// snn_core_pipe.v — Synthesisable pipelined FSM implementation of the SNN crossbar core.
//
// One multiply-accumulate operation per clock cycle.  The behavioral model
// (snn_core_fixed.v) completes all timesteps in a single simulation cycle and
// uses $readmemh — neither is synthesisable.  This module replaces both with:
//   - A registered write port for weight loading (wr_en / wr_sel / wr_addr / wr_data)
//   - A flat spike-input bus presented at start
//   - A 4-state FSM that serialises the MAC loops
//
// Total inference latency:
//   NUM_STEPS * (HIDDEN_DIM * INPUT_DIM + OUTPUT_DIM * HIDDEN_DIM) cycles
//
// Arithmetic:
//   Decay  : (BETA_NUM * mem) >>> BETA_SHIFT  (arithmetic right-shift = floor div
//            for signed values when BETA_DEN = 2^BETA_SHIFT)
//   Weights: signed INT8
//   Membrane: signed INT32
//   Threshold: configurable (default 128)
//
// Weight write port:
//   wr_en=1, wr_sel=0 -> w1[wr_addr] = wr_data  (hidden-layer weights)
//   wr_en=1, wr_sel=1 -> w2[wr_addr] = wr_data  (output-layer weights)
//   Write port is only sampled when state == IDLE (safe to write between inferences).
//
// Output:
//   logits_packed [OUTPUT_DIM*32-1:0]: logit[o] occupies bits [o*32 +: 32], valid when done=1.

`timescale 1ns/1ps

module snn_core_pipe #(
    parameter integer INPUT_DIM  = 784,
    parameter integer HIDDEN_DIM = 128,
    parameter integer OUTPUT_DIM = 10,
    parameter integer NUM_STEPS  = 10,
    parameter integer BETA_NUM   = 983,
    parameter integer BETA_SHIFT = 10,      // BETA_DEN = 2^BETA_SHIFT = 1024
    parameter integer THRESHOLD  = 128,
    parameter integer WR_ADDR_W  = 17       // ceil(log2(HIDDEN_DIM * INPUT_DIM))
)(
    input  wire clk,
    input  wire rst_n,
    input  wire start,
    output reg  done,

    // Spike inputs — flat bitmap, stable from start until done.
    // Bit [t*INPUT_DIM + i] = spike at timestep t, input channel i.
    input  wire [NUM_STEPS*INPUT_DIM-1:0] spk_in,

    // Weight write port (active only while state == IDLE).
    input  wire                    wr_en,
    input  wire                    wr_sel,          // 0 = w1, 1 = w2
    input  wire [WR_ADDR_W-1:0]    wr_addr,
    input  wire signed [7:0]       wr_data,

    // Packed logit output — logit[o] = logits_packed[o*32 +: 32]
    output wire [OUTPUT_DIM*32-1:0] logits_packed
);

    // -----------------------------------------------------------------------
    // Local parameters
    // -----------------------------------------------------------------------
    localparam integer W1_SIZE = HIDDEN_DIM * INPUT_DIM;
    localparam integer W2_SIZE = OUTPUT_DIM * HIDDEN_DIM;

    // Counter widths — add 1 bit as carry guard
    localparam integer T_W   = $clog2(NUM_STEPS)  + 1;
    localparam integer H_W   = $clog2(HIDDEN_DIM) + 1;
    localparam integer O_W   = $clog2(OUTPUT_DIM) + 1;
    localparam integer IDX_W = $clog2(INPUT_DIM)  + 1;  // INPUT_DIM >= HIDDEN_DIM

    // FSM states
    localparam [1:0] IDLE   = 2'd0,
                     STEP_H = 2'd1,
                     STEP_O = 2'd2,
                     S_DONE = 2'd3;

    // -----------------------------------------------------------------------
    // Weight memories (infer SRAM or flop arrays depending on target)
    // -----------------------------------------------------------------------
    reg signed [7:0] w1 [0:W1_SIZE-1];
    reg signed [7:0] w2 [0:W2_SIZE-1];

    // -----------------------------------------------------------------------
    // State and datapath registers
    // -----------------------------------------------------------------------
    reg [1:0]  state;
    reg [T_W-1:0]   t_cnt;      // current timestep
    reg [H_W-1:0]   h_cnt;      // current hidden neuron (STEP_H) or inner h loop (STEP_O)
    reg [O_W-1:0]   o_cnt;      // current output neuron (STEP_O)
    reg [IDX_W-1:0] idx_cnt;    // inner MAC loop counter

    reg signed [31:0] mem1 [0:HIDDEN_DIM-1];
    reg signed [31:0] mem2 [0:OUTPUT_DIM-1];
    reg               spk1 [0:HIDDEN_DIM-1];
    reg signed [31:0] logits [0:OUTPUT_DIM-1];

    reg signed [63:0] cur_acc;  // running MAC accumulator (signed 64-bit)

    // -----------------------------------------------------------------------
    // Logit output pack
    // -----------------------------------------------------------------------
    genvar gv;
    generate
        for (gv = 0; gv < OUTPUT_DIM; gv = gv + 1) begin : PACK_LOGITS
            assign logits_packed[gv*32 +: 32] = logits[gv];
        end
    endgenerate

    // -----------------------------------------------------------------------
    // Weight write port (only in IDLE)
    // -----------------------------------------------------------------------
    integer wi;
    always @(posedge clk) begin
        if (state == IDLE && wr_en) begin
            if (!wr_sel)
                w1[wr_addr] <= wr_data;
            else
                w2[wr_addr] <= wr_data;
        end
    end

    // -----------------------------------------------------------------------
    // FSM + datapath
    // -----------------------------------------------------------------------
    // Intermediate wires for finalise steps (computed combinatorially)
    reg signed [63:0] mem_pre;   // used as temporary inside always block

    integer ri, rh, ro;  // reset loop variables

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state   <= IDLE;
            done    <= 1'b0;
            t_cnt   <= 0;
            h_cnt   <= 0;
            o_cnt   <= 0;
            idx_cnt <= 0;
            cur_acc <= 0;
            for (ri = 0; ri < HIDDEN_DIM; ri = ri + 1) begin
                mem1[ri] <= 0;
                spk1[ri] <= 1'b0;
            end
            for (ro = 0; ro < OUTPUT_DIM; ro = ro + 1) begin
                mem2[ro]   <= 0;
                logits[ro] <= 0;
            end
        end else begin
            done <= 1'b0;

            case (state)
                // ----------------------------------------------------------
                IDLE: begin
                    if (start) begin
                        // Clear state for new inference
                        for (ri = 0; ri < HIDDEN_DIM; ri = ri + 1) begin
                            mem1[ri] <= 0;
                            spk1[ri] <= 1'b0;
                        end
                        for (ro = 0; ro < OUTPUT_DIM; ro = ro + 1) begin
                            mem2[ro]   <= 0;
                            logits[ro] <= 0;
                        end
                        t_cnt   <= 0;
                        h_cnt   <= 0;
                        idx_cnt <= 0;
                        cur_acc <= 0;
                        state   <= STEP_H;
                    end
                end

                // ----------------------------------------------------------
                // STEP_H: MAC loop over inputs for hidden neuron h_cnt.
                //   idx_cnt iterates 0..INPUT_DIM-1.
                //   When idx_cnt reaches INPUT_DIM-1, finalise and advance h_cnt.
                // ----------------------------------------------------------
                STEP_H: begin
                    // Accumulate spike contribution for input idx_cnt
                    if (spk_in[t_cnt * INPUT_DIM + idx_cnt])
                        cur_acc <= cur_acc + $signed({{56{w1[h_cnt * INPUT_DIM + idx_cnt][7]}},
                                                      w1[h_cnt * INPUT_DIM + idx_cnt]});

                    if (idx_cnt == INPUT_DIM - 1) begin
                        // Finalise hidden neuron h_cnt
                        // mem_pre = floor((BETA_NUM * mem1[h_cnt]) / 2^BETA_SHIFT) + cur_acc
                        // (cur_acc at this point doesn't include the current spike yet —
                        //  include it inline)
                        mem_pre = (($signed(BETA_NUM) * $signed(mem1[h_cnt])) >>> BETA_SHIFT)
                                  + cur_acc
                                  + (spk_in[t_cnt * INPUT_DIM + idx_cnt] ?
                                     $signed({{56{w1[h_cnt * INPUT_DIM + idx_cnt][7]}},
                                              w1[h_cnt * INPUT_DIM + idx_cnt]}) : 64'sd0);
                        if (mem_pre >= $signed(THRESHOLD)) begin
                            spk1[h_cnt] <= 1'b1;
                            mem1[h_cnt] <= mem_pre[31:0] - THRESHOLD;
                        end else begin
                            spk1[h_cnt] <= 1'b0;
                            mem1[h_cnt] <= mem_pre[31:0];
                        end
                        cur_acc <= 0;

                        if (h_cnt == HIDDEN_DIM - 1) begin
                            // All hidden neurons done for this timestep → output phase
                            h_cnt   <= 0;   // reuse as inner loop var for STEP_O
                            idx_cnt <= 0;
                            o_cnt   <= 0;
                            state   <= STEP_O;
                        end else begin
                            h_cnt   <= h_cnt + 1;
                            idx_cnt <= 0;
                        end
                    end else begin
                        idx_cnt <= idx_cnt + 1;
                    end
                end

                // ----------------------------------------------------------
                // STEP_O: MAC loop over hidden neurons for output neuron o_cnt.
                //   idx_cnt iterates 0..HIDDEN_DIM-1.
                //   When idx_cnt reaches HIDDEN_DIM-1, finalise and advance o_cnt.
                // ----------------------------------------------------------
                STEP_O: begin
                    // Accumulate spike contribution for hidden idx_cnt
                    if (spk1[idx_cnt])
                        cur_acc <= cur_acc + $signed({{56{w2[o_cnt * HIDDEN_DIM + idx_cnt][7]}},
                                                      w2[o_cnt * HIDDEN_DIM + idx_cnt]});

                    if (idx_cnt == HIDDEN_DIM - 1) begin
                        // Finalise output neuron o_cnt
                        mem_pre = (($signed(BETA_NUM) * $signed(mem2[o_cnt])) >>> BETA_SHIFT)
                                  + cur_acc
                                  + (spk1[idx_cnt] ?
                                     $signed({{56{w2[o_cnt * HIDDEN_DIM + idx_cnt][7]}},
                                              w2[o_cnt * HIDDEN_DIM + idx_cnt]}) : 64'sd0);
                        if (mem_pre >= $signed(THRESHOLD)) begin
                            logits[o_cnt] <= logits[o_cnt] + 1;
                            mem2[o_cnt]   <= mem_pre[31:0] - THRESHOLD;
                        end else begin
                            mem2[o_cnt] <= mem_pre[31:0];
                        end
                        cur_acc <= 0;

                        if (o_cnt == OUTPUT_DIM - 1) begin
                            if (t_cnt == NUM_STEPS - 1) begin
                                state <= S_DONE;
                            end else begin
                                // Next timestep
                                t_cnt   <= t_cnt + 1;
                                h_cnt   <= 0;
                                idx_cnt <= 0;
                                o_cnt   <= 0;
                                state   <= STEP_H;
                            end
                        end else begin
                            o_cnt   <= o_cnt + 1;
                            idx_cnt <= 0;
                        end
                    end else begin
                        idx_cnt <= idx_cnt + 1;
                    end
                end

                // ----------------------------------------------------------
                S_DONE: begin
                    done  <= 1'b1;
                    state <= IDLE;
                end

                default: state <= IDLE;
            endcase
        end
    end

endmodule
