`timescale 1ns/1ps

module snn_core_fixed #(
    parameter integer INPUT_DIM = 784,
    parameter integer HIDDEN_DIM = 128,
    parameter integer OUTPUT_DIM = 10,
    parameter integer NUM_STEPS = 10,
    parameter integer BETA_NUM = 983,
    parameter integer BETA_DEN = 1024,
    parameter integer THRESHOLD = 128
)(
    input  wire clk,
    input  wire rst_n,
    input  wire start,
    output reg  done,
    // High whenever the core is not idle: running timesteps or in sticky-done.
    output wire busy
);

    localparam integer W1_SIZE = HIDDEN_DIM * INPUT_DIM;
    localparam integer W2_SIZE = OUTPUT_DIM * HIDDEN_DIM;
    localparam integer S_SIZE  = NUM_STEPS * INPUT_DIM;

    reg signed [7:0] w1 [0:W1_SIZE-1];
    reg signed [7:0] w2 [0:W2_SIZE-1];
    reg               spikes [0:S_SIZE-1];

    reg signed [31:0] mem1 [0:HIDDEN_DIM-1];
    reg signed [31:0] mem2 [0:OUTPUT_DIM-1];
    reg               spk1 [0:HIDDEN_DIM-1];
    reg signed [31:0] logits [0:OUTPUT_DIM-1];

    localparam [1:0] S_IDLE = 2'd0;
    localparam [1:0] S_RUN  = 2'd1;
    localparam [1:0] S_DONE = 2'd2;

    reg [1:0] state = S_IDLE;
    reg [31:0] t_step = 0;
    reg start_prev = 1'b0;
    reg need_start_low = 1'b0;

    assign busy = rst_n && (state != S_IDLE);

    integer i, h, o;
    integer idx;
    reg signed [63:0] cur1;
    reg signed [63:0] cur2;
    reg signed [63:0] mem_pre;

    // Floor division matching Python // semantics (round toward -inf).
    // C/Verilog / truncates toward zero; for negative numerators with a
    // positive denominator (always the case for BETA_DEN) the results differ.
    function automatic signed [63:0] floor_div_64(
        input signed [63:0] a,
        input signed [63:0] b
    );
        reg signed [63:0] q, r;
        begin
            q = a / b;
            r = a % b;
            // When remainder is non-zero and signs of a and b differ, subtract 1.
            floor_div_64 = (r != 0 && (a[63] ^ b[63])) ? q - 1 : q;
        end
    endfunction

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            done <= 1'b0;
            state <= S_IDLE;
            t_step <= 0;
            start_prev <= 1'b0;
            need_start_low <= 1'b0;
            for (h = 0; h < HIDDEN_DIM; h = h + 1) begin
                mem1[h] <= 0;
                spk1[h] <= 1'b0;
            end
            for (o = 0; o < OUTPUT_DIM; o = o + 1) begin
                mem2[o] <= 0;
                logits[o] <= 0;
            end
        end else begin
            case (state)
                S_IDLE: begin
                    done <= 1'b0;
                    if (!start) begin
                        need_start_low <= 1'b0;
                    end
                    // Arm new jobs on a rising edge of start, except right after a sticky-done
                    // completion where we require start to return low before re-arming.
                    if (start && !start_prev && !need_start_low) begin
                        t_step <= 0;
                        for (h = 0; h < HIDDEN_DIM; h = h + 1) begin
                            mem1[h] = 0;
                            spk1[h] = 1'b0;
                        end
                        for (o = 0; o < OUTPUT_DIM; o = o + 1) begin
                            mem2[o] = 0;
                            logits[o] = 0;
                        end
                        state <= S_RUN;
                    end
                end
                S_RUN: begin
                    done <= 1'b0;
                    for (h = 0; h < HIDDEN_DIM; h = h + 1) begin
                        cur1 = 0;
                        for (i = 0; i < INPUT_DIM; i = i + 1) begin
                            idx = t_step * INPUT_DIM + i;
                            if (spikes[idx]) begin
                                cur1 = cur1 + $signed(w1[h * INPUT_DIM + i]);
                            end
                        end
                        mem_pre = floor_div_64($signed(BETA_NUM) * $signed(mem1[h]), $signed(BETA_DEN)) + cur1;
                        if (mem_pre >= THRESHOLD) begin
                            spk1[h] = 1'b1;
                            mem1[h] = mem_pre - THRESHOLD;
                        end else begin
                            spk1[h] = 1'b0;
                            mem1[h] = mem_pre;
                        end
                    end

                    for (o = 0; o < OUTPUT_DIM; o = o + 1) begin
                        cur2 = 0;
                        for (h = 0; h < HIDDEN_DIM; h = h + 1) begin
                            if (spk1[h]) begin
                                cur2 = cur2 + $signed(w2[o * HIDDEN_DIM + h]);
                            end
                        end
                        mem_pre = floor_div_64($signed(BETA_NUM) * $signed(mem2[o]), $signed(BETA_DEN)) + cur2;
                        if (mem_pre >= THRESHOLD) begin
                            mem2[o] = mem_pre - THRESHOLD;
                            logits[o] = logits[o] + 1;
                        end else begin
                            mem2[o] = mem_pre;
                        end
                    end

                    if (t_step == (NUM_STEPS - 1)) begin
                        state <= S_DONE;
                    end else begin
                        t_step <= t_step + 1;
                    end
                end
                S_DONE: begin
                    // Sticky done: hold completion until start is deasserted.
                    done <= 1'b1;
                    if (!start) begin
                        need_start_low <= 1'b1;
                        state <= S_IDLE;
                    end
                end
                default: begin
                    done <= 1'b0;
                    state <= S_IDLE;
                end
            endcase

            start_prev <= start;
        end
    end

endmodule
