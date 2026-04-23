// SystemC fixed-point SNN reference model.
//
// Implements the same 2-layer LIF network as:
//   - Python:   scripts/run_rtl_reference_check.py  (run_python_fixed)
//   - C++:      ref/cpp/crossbar_snn_ref_fixed.cpp
//   - Verilog:  src/snn_core_fixed.v
//
// The module interface mirrors snn_core_fixed.v exactly (clk, rst_n, start,
// done) so it can later be used as a TLM reference in a mixed-language flow.
//
// Compile:
//   g++ -O2 -std=c++17 -I/usr/include \
//       ref/systemc/crossbar_snn_ref_fixed_sc.cpp -lsystemc \
//       -o <out_bin>
//
// Usage (identical argument layout to crossbar_snn_ref_fixed):
//   <bin> <config_fixed.txt> <w1.memh> <w2.memh> <spikes.memh> \
//         <expected_logits.txt> <output_logits.txt> <summary.txt>

#include <systemc.h>

#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Arithmetic helpers (identical to crossbar_snn_ref_fixed.cpp)
// ---------------------------------------------------------------------------

namespace {

// Match Python // (floor division): round toward -inf, not toward zero.
inline int64_t floor_div(int64_t a, int64_t b) {
    int64_t q = a / b;
    int64_t r = a % b;
    return (r != 0 && ((a ^ b) < 0)) ? q - 1 : q;
}

int32_t sign_extend(uint32_t x, int width_bits) {
    const uint32_t mask = (width_bits >= 32) ? 0xFFFFFFFFU : ((1U << width_bits) - 1U);
    x &= mask;
    const uint32_t sign_bit = 1U << (width_bits - 1);
    if (x & sign_bit)
        return static_cast<int32_t>(x | (~mask));
    return static_cast<int32_t>(x);
}

std::vector<int32_t> read_memh_signed_variable(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Failed to open: " + path);
    std::vector<int32_t> values;
    std::string tok;
    while (in >> tok) {
        uint32_t raw = 0;
        std::stringstream ss;
        ss << std::hex << tok;
        ss >> raw;
        int width_bits = 0;
        if      (tok.size() <= 2) width_bits = 8;
        else if (tok.size() <= 4) width_bits = 16;
        else if (tok.size() <= 8) width_bits = 32;
        else throw std::runtime_error("Unsupported memh token width: " + tok);
        values.push_back(sign_extend(raw, width_bits));
    }
    return values;
}

std::vector<int32_t> read_memh_bit(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Failed to open: " + path);
    std::vector<int32_t> values;
    std::string tok;
    while (in >> tok) values.push_back((tok == "1") ? 1 : 0);
    return values;
}

std::vector<int32_t> read_dec_vector(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Failed to open: " + path);
    std::vector<int32_t> values;
    int32_t v = 0;
    while (in >> v) values.push_back(v);
    return values;
}

}  // namespace

// ---------------------------------------------------------------------------
// SC_MODULE: SnnCoreFixed
//
// Ports mirror snn_core_fixed.v.  Weight and spike vectors are passed in at
// construction time (read from files before elaboration in sc_main).
//
// Computation semantics:
//   On the first posedge clk after start is asserted, all num_steps timesteps
//   are executed in a single simulation cycle (matching the Verilog behavioral
//   model).  done is asserted for that same cycle.
// ---------------------------------------------------------------------------

SC_MODULE(SnnCoreFixed) {
    sc_in_clk       clk;
    sc_in<bool>     rst_n;
    sc_in<bool>     start;
    sc_out<bool>    done;

    // Results accessible after done is asserted.
    std::vector<int32_t> logits;

    SC_HAS_PROCESS(SnnCoreFixed);

    SnnCoreFixed(sc_module_name name,
                 int input_dim, int hidden_dim, int output_dim, int num_steps,
                 int beta_num,  int beta_den,   int threshold,
                 std::vector<int32_t> w1,
                 std::vector<int32_t> w2,
                 std::vector<int32_t> spikes)
        : sc_module(name),
          logits(output_dim, 0),
          m_input_dim(input_dim), m_hidden_dim(hidden_dim),
          m_output_dim(output_dim), m_num_steps(num_steps),
          m_beta_num(beta_num), m_beta_den(beta_den), m_threshold(threshold),
          m_w1(std::move(w1)), m_w2(std::move(w2)), m_spikes(std::move(spikes))
    {
        SC_CTHREAD(compute, clk.pos());
    }

    void compute() {
        while (true) {
            wait();  // posedge clk

            if (!rst_n.read()) {
                done.write(false);
                std::fill(logits.begin(), logits.end(), 0);
                continue;
            }

            done.write(false);

            if (!start.read())
                continue;

            // --- Run all timesteps (single-cycle behavioral, matches Verilog) ---
            std::vector<int32_t> mem1(m_hidden_dim, 0);
            std::vector<int32_t> mem2(m_output_dim, 0);
            std::vector<int32_t> spk1(m_hidden_dim, 0);
            std::fill(logits.begin(), logits.end(), 0);

            for (int t = 0; t < m_num_steps; ++t) {
                for (int h = 0; h < m_hidden_dim; ++h) {
                    int64_t cur1 = 0;
                    for (int i = 0; i < m_input_dim; ++i) {
                        if (m_spikes[t * m_input_dim + i])
                            cur1 += m_w1[h * m_input_dim + i];
                    }
                    int32_t mem_pre = static_cast<int32_t>(
                        floor_div(static_cast<int64_t>(m_beta_num) * mem1[h], m_beta_den) + cur1);
                    if (mem_pre >= m_threshold) {
                        spk1[h] = 1;
                        mem1[h] = mem_pre - m_threshold;
                    } else {
                        spk1[h] = 0;
                        mem1[h] = mem_pre;
                    }
                }

                for (int o = 0; o < m_output_dim; ++o) {
                    int64_t cur2 = 0;
                    for (int h = 0; h < m_hidden_dim; ++h) {
                        if (spk1[h])
                            cur2 += m_w2[o * m_hidden_dim + h];
                    }
                    int32_t mem_pre = static_cast<int32_t>(
                        floor_div(static_cast<int64_t>(m_beta_num) * mem2[o], m_beta_den) + cur2);
                    if (mem_pre >= m_threshold) {
                        mem2[o] = mem_pre - m_threshold;
                        logits[o] += 1;
                    } else {
                        mem2[o] = mem_pre;
                    }
                }
            }

            done.write(true);
        }
    }

private:
    int m_input_dim, m_hidden_dim, m_output_dim, m_num_steps;
    int m_beta_num, m_beta_den, m_threshold;
    std::vector<int32_t> m_w1, m_w2, m_spikes;
};

// ---------------------------------------------------------------------------
// sc_main: parse args, build DUT, run simulation, verify, write outputs
// ---------------------------------------------------------------------------

int sc_main(int argc, char* argv[]) {
    if (argc != 8) {
        std::cerr << "Usage: " << argv[0]
                  << " <config_fixed.txt> <w1.memh> <w2.memh> <spikes.memh>"
                     " <expected_logits.txt> <output_logits.txt> <summary.txt>\n";
        return 2;
    }

    try {
        // --- Config ---
        std::ifstream cfg(argv[1]);
        if (!cfg) throw std::runtime_error("Failed to open config: " + std::string(argv[1]));
        int input_dim = 0, hidden_dim = 0, output_dim = 0, num_steps = 0;
        int beta_num = 0, beta_den = 1, threshold = 1;
        cfg >> input_dim >> hidden_dim >> output_dim >> num_steps
            >> beta_num >> beta_den >> threshold;
        if (!cfg) throw std::runtime_error("Invalid config_fixed values");
        if (beta_den == 0) throw std::runtime_error("beta_den must not be zero");

        // --- Vectors ---
        const auto w1      = read_memh_signed_variable(argv[2]);
        const auto w2      = read_memh_signed_variable(argv[3]);
        const auto spikes  = read_memh_bit(argv[4]);
        const auto expected = read_dec_vector(argv[5]);

        if (static_cast<int>(w1.size())     != hidden_dim * input_dim)
            throw std::runtime_error("w1 size mismatch");
        if (static_cast<int>(w2.size())     != output_dim * hidden_dim)
            throw std::runtime_error("w2 size mismatch");
        if (static_cast<int>(spikes.size()) != num_steps * input_dim)
            throw std::runtime_error("spikes size mismatch");
        if (static_cast<int>(expected.size()) != output_dim)
            throw std::runtime_error("expected size mismatch");

        // --- Signals ---
        sc_clock      clk("clk", 10, SC_NS);
        sc_signal<bool> rst_n("rst_n"), start("start"), done("done");

        // --- DUT ---
        SnnCoreFixed dut("dut",
                         input_dim, hidden_dim, output_dim, num_steps,
                         beta_num, beta_den, threshold,
                         w1, w2, spikes);
        dut.clk(clk);
        dut.rst_n(rst_n);
        dut.start(start);
        dut.done(done);

        // --- Stimulus: reset → start pulse → wait for done ---
        rst_n.write(false);
        start.write(false);
        sc_start(20, SC_NS);   // two cycles in reset

        rst_n.write(true);
        sc_start(10, SC_NS);   // one cycle released

        start.write(true);
        sc_start(10, SC_NS);   // one cycle with start high
        start.write(false);

        // Poll for done with a 64-cycle timeout (computation is single-cycle).
        int timeout = 64;
        while (!done.read() && timeout-- > 0)
            sc_start(10, SC_NS);

        if (!done.read())
            throw std::runtime_error("Timeout: done not asserted within 64 cycles");

        // --- Verify and write outputs ---
        const std::vector<int32_t>& logits = dut.logits;

        int max_abs_err = 0;
        for (int o = 0; o < output_dim; ++o) {
            int err = std::abs(logits[o] - expected[o]);
            if (err > max_abs_err) max_abs_err = err;
        }

        std::ofstream out(argv[6]);
        if (!out) throw std::runtime_error("Failed to open output logits file");
        for (int o = 0; o < output_dim; ++o) out << logits[o] << "\n";

        std::ofstream summary(argv[7]);
        if (!summary) throw std::runtime_error("Failed to open summary file");
        summary << "max_abs_err=" << max_abs_err << "\n";

        std::cout << "SystemC reference complete. max_abs_err=" << max_abs_err << "\n";
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}
