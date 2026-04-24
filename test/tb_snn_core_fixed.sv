`timescale 1ns/1ps

`ifndef INPUT_DIM
`define INPUT_DIM 784
`endif
`ifndef HIDDEN_DIM
`define HIDDEN_DIM 128
`endif
`ifndef OUTPUT_DIM
`define OUTPUT_DIM 10
`endif
`ifndef NUM_STEPS
`define NUM_STEPS 10
`endif
`ifndef BETA_NUM
`define BETA_NUM 983
`endif
`ifndef BETA_DEN
`define BETA_DEN 1024
`endif
`ifndef THRESHOLD
`define THRESHOLD 128
`endif

module tb_snn_core_fixed;
    reg clk = 1'b0;
    reg rst_n = 1'b0;
    reg start = 1'b0;
    wire done;
    wire busy;
    integer cycle_ctr = 0;
    integer start_cycle = -1;
    integer done_cycle = -1;
    integer latency_cycles;

    integer fd;
    integer o;
    integer exp_logits [0:`OUTPUT_DIM-1];

    snn_core_fixed #(
        .INPUT_DIM(`INPUT_DIM),
        .HIDDEN_DIM(`HIDDEN_DIM),
        .OUTPUT_DIM(`OUTPUT_DIM),
        .NUM_STEPS(`NUM_STEPS),
        .BETA_NUM(`BETA_NUM),
        .BETA_DEN(`BETA_DEN),
        .THRESHOLD(`THRESHOLD),
        .W1_FILE("artifacts/ref_vectors_fixed/w1.memh"),
        .W2_FILE("artifacts/ref_vectors_fixed/w2.memh"),
        .SPIKES_FILE("artifacts/ref_vectors_fixed/spikes.memh")
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .done(done),
        .busy(busy)
    );

    always #5 clk = ~clk;
    always @(posedge clk) begin
        if (rst_n) cycle_ctr <= cycle_ctr + 1;
    end

    task automatic apply_reset;
        begin
            start = 1'b0;
            rst_n = 1'b0;
            repeat (2) @(posedge clk);
            rst_n = 1'b1;
            @(posedge clk);
        end
    endtask

    task automatic issue_start_pulse_one_cycle;
        begin
            start = 1'b1;
            @(posedge clk);
            start = 1'b0;
        end
    endtask

    task automatic wait_done_with_timeout(input integer timeout_cycles, input [8*64-1:0] msg);
        integer k;
        reg saw_done;
        begin
            saw_done = 1'b0;
            for (k = 0; k < timeout_cycles; k = k + 1) begin
                @(posedge clk);
                if (done == 1'b1) begin
                    saw_done = 1'b1;
                    k = timeout_cycles;
                end
            end
            if (!saw_done) begin
                $fatal(1, "%0s", msg);
            end
        end
    endtask

    task automatic assert_latency(input integer expected_cycles, input [8*64-1:0] msg);
        begin
            done_cycle = cycle_ctr;
            latency_cycles = done_cycle - start_cycle;
            if (latency_cycles != expected_cycles) begin
                $fatal(1, "%0s: expected %0d cycles, got %0d",
                       msg, expected_cycles, latency_cycles);
            end
        end
    endtask

    task automatic capture_expected_logits;
        begin
            for (o = 0; o < `OUTPUT_DIM; o = o + 1) begin
                exp_logits[o] = dut.logits[o];
            end
        end
    endtask

    task automatic assert_logits_match_expected(input [8*64-1:0] msg);
        begin
            for (o = 0; o < `OUTPUT_DIM; o = o + 1) begin
                if (dut.logits[o] !== exp_logits[o]) begin
                    $fatal(1, "%0s: mismatch at o=%0d exp=%0d got=%0d",
                           msg, o, exp_logits[o], dut.logits[o]);
                end
            end
        end
    endtask

    task automatic assert_done_low_for_cycles(input integer n, input [8*64-1:0] msg);
        integer k;
        begin
            for (k = 0; k < n; k = k + 1) begin
                @(posedge clk);
                if (done !== 1'b0) begin
                    $fatal(1, "%0s: done asserted early (k=%0d)", msg, k);
                end
            end
        end
    endtask

    task automatic write_logits_file;
        begin
            fd = $fopen("artifacts/ref_vectors_fixed/verilog_logits.txt", "w");
            if (fd == 0) begin
                $display("ERROR: failed to open output file");
                $finish(1);
            end
            for (o = 0; o < `OUTPUT_DIM; o = o + 1) begin
                $fwrite(fd, "%0d\n", dut.logits[o]);
            end
            $fclose(fd);
            $display("Wrote artifacts/ref_vectors_fixed/verilog_logits.txt");
        end
    endtask

    initial begin
        #20; apply_reset();

        // Scenario 1: one-cycle start pulse, verify fixed latency to done.
        issue_start_pulse_one_cycle();
        start_cycle = cycle_ctr;
        wait_done_with_timeout(`NUM_STEPS + 16, "TIMEOUT: done not asserted within expected multi-cycle window");
        assert_latency(`NUM_STEPS + 1, "LATENCY_MISMATCH");
        capture_expected_logits();

        // Scenario 2: hold start high through completion; done must stick high
        // until start is released.
        apply_reset();

        start = 1'b1;
        @(posedge clk);
        start_cycle = cycle_ctr;
        wait_done_with_timeout(`NUM_STEPS + 16, "TIMEOUT_HOLD_START: done not asserted while start held high");
        assert_latency(`NUM_STEPS + 1, "LATENCY_MISMATCH_HOLD_START");

        repeat (3) begin
            @(posedge clk);
            if (done !== 1'b1) begin
                $fatal(1, "DONE_NOT_STICKY: done deasserted while start remained high");
            end
        end

        start = 1'b0;
        @(posedge clk);
        if (done !== 1'b1) begin
            $fatal(1, "DONE_CLEAR_PHASE_ERR: expected done high during release cycle");
        end
        @(posedge clk);
        if (done !== 1'b0) begin
            $fatal(1, "DONE_NOT_CLEARED: done remained high after release-to-idle transition");
        end

        // Scenario 3: back-to-back run without reset; logits must match scenario 1.
        if (start !== 1'b0 || done !== 1'b0) begin
            $fatal(1, "PRECOND_FAIL: expected idle between scenarios");
        end
        issue_start_pulse_one_cycle();
        start_cycle = cycle_ctr;
        wait_done_with_timeout(`NUM_STEPS + 16, "TIMEOUT_BACK2BACK: done not asserted");
        assert_latency(`NUM_STEPS + 1, "LATENCY_MISMATCH_BACK2BACK");
        assert_logits_match_expected("LOGIT_MISMATCH_BACK2BACK");

        // Scenario 4: after sticky-done release, a start line that remains asserted
        // without a fresh low->high transition must not re-arm a new run.
        //
        // Note: the DUT may set an internal "need start low" flag when exiting sticky
        // done; if `start` is already high from a prior phase, re-entering the hold
        // scenario requires a quiescent low period (here: reset) so the next rising
        // edge is unambiguous.
        apply_reset();

        start = 1'b1;
        @(posedge clk);
        start_cycle = cycle_ctr;
        wait_done_with_timeout(`NUM_STEPS + 16, "TIMEOUT_REARM_HOLD: done not asserted while start held high");
        assert_latency(`NUM_STEPS + 1, "LATENCY_MISMATCH_REARM_HOLD");

        repeat (3) begin
            @(posedge clk);
            if (done !== 1'b1) begin
                $fatal(1, "DONE_NOT_STICKY_REARM: done deasserted while start remained high");
            end
        end

        // Release sticky done, then return start high *without* an intervening edge.
        start = 1'b0;
        @(posedge clk);
        if (done !== 1'b1) begin
            $fatal(1, "DONE_CLEAR_PHASE_ERR_REARM: expected done high during release cycle");
        end
        @(posedge clk);
        if (done !== 1'b0) begin
            $fatal(1, "DONE_NOT_CLEARED_REARM: done remained high after release-to-idle transition");
        end

        start = 1'b1;
        @(posedge clk);
        // If a new run incorrectly arms here, done typically asserts ~NUM_STEPS+1 cycles
        // after the accepted start edge. Stay safely below that window.
        assert_done_low_for_cycles(`NUM_STEPS, "REARM_HOLD_SHOULD_NOT_START");

        write_logits_file();
        $finish;
    end
endmodule
