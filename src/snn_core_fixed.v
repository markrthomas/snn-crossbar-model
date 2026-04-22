`timescale 1ns/1ps

module snn_core_fixed #(
    parameter integer INPUT_DIM = 784,
    parameter integer HIDDEN_DIM = 128,
    parameter integer OUTPUT_DIM = 10,
    parameter integer NUM_STEPS = 10,
    parameter integer BETA_NUM = 95,
    parameter integer BETA_DEN = 100,
    parameter integer THRESHOLD = 256,
    parameter string W1_FILE = "artifacts/ref_vectors_fixed/w1.memh",
    parameter string W2_FILE = "artifacts/ref_vectors_fixed/w2.memh",
    parameter string SPIKES_FILE = "artifacts/ref_vectors_fixed/spikes.memh"
)(
    input  wire clk,
    input  wire rst_n,
    input  wire start,
    output reg  done
);

    localparam integer W1_SIZE = HIDDEN_DIM * INPUT_DIM;
    localparam integer W2_SIZE = OUTPUT_DIM * HIDDEN_DIM;
    localparam integer S_SIZE  = NUM_STEPS * INPUT_DIM;

    reg signed [15:0] w1 [0:W1_SIZE-1];
    reg signed [15:0] w2 [0:W2_SIZE-1];
    reg               spikes [0:S_SIZE-1];

    reg signed [31:0] mem1 [0:HIDDEN_DIM-1];
    reg signed [31:0] mem2 [0:OUTPUT_DIM-1];
    reg               spk1 [0:HIDDEN_DIM-1];
    reg signed [31:0] logits [0:OUTPUT_DIM-1];

    integer t, i, h, o;
    integer idx;
    reg signed [63:0] cur1;
    reg signed [63:0] cur2;
    reg signed [63:0] mem_pre;

    initial begin
        $readmemh(W1_FILE, w1);
        $readmemh(W2_FILE, w2);
        $readmemh(SPIKES_FILE, spikes);
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            done <= 1'b0;
            for (h = 0; h < HIDDEN_DIM; h = h + 1) begin
                mem1[h] <= 0;
                spk1[h] <= 1'b0;
            end
            for (o = 0; o < OUTPUT_DIM; o = o + 1) begin
                mem2[o] <= 0;
                logits[o] <= 0;
            end
        end else begin
            done <= 1'b0;
            if (start) begin
                for (h = 0; h < HIDDEN_DIM; h = h + 1) begin
                    mem1[h] = 0;
                    spk1[h] = 1'b0;
                end
                for (o = 0; o < OUTPUT_DIM; o = o + 1) begin
                    mem2[o] = 0;
                    logits[o] = 0;
                end

                for (t = 0; t < NUM_STEPS; t = t + 1) begin
                    for (h = 0; h < HIDDEN_DIM; h = h + 1) begin
                        cur1 = 0;
                        for (i = 0; i < INPUT_DIM; i = i + 1) begin
                            idx = t * INPUT_DIM + i;
                            if (spikes[idx]) begin
                                cur1 = cur1 + $signed(w1[h * INPUT_DIM + i]);
                            end
                        end
                        mem_pre = (($signed(BETA_NUM) * $signed(mem1[h])) / $signed(BETA_DEN)) + cur1;
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
                        mem_pre = (($signed(BETA_NUM) * $signed(mem2[o])) / $signed(BETA_DEN)) + cur2;
                        if (mem_pre >= THRESHOLD) begin
                            mem2[o] = mem_pre - THRESHOLD;
                            logits[o] = logits[o] + 1;
                        end else begin
                            mem2[o] = mem_pre;
                        end
                    end
                end
                done <= 1'b1;
            end
        end
    end

endmodule
