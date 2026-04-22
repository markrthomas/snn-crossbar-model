#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

int32_t sign_extend16(uint32_t x) {
    return (x & 0x8000U) ? static_cast<int32_t>(x | 0xFFFF0000U) : static_cast<int32_t>(x);
}

std::vector<int32_t> read_memh_signed16(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Failed to open: " + path);
    std::vector<int32_t> values;
    std::string tok;
    while (in >> tok) {
        uint32_t raw = 0;
        std::stringstream ss;
        ss << std::hex << tok;
        ss >> raw;
        values.push_back(sign_extend16(raw & 0xFFFFU));
    }
    return values;
}

std::vector<int32_t> read_memh_bit(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Failed to open: " + path);
    std::vector<int32_t> values;
    std::string tok;
    while (in >> tok) {
        values.push_back((tok == "1") ? 1 : 0);
    }
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

int main(int argc, char** argv) {
    if (argc != 8) {
        std::cerr << "Usage: " << argv[0]
                  << " <config_fixed.txt> <w1.memh> <w2.memh> <spikes.memh> <expected_logits.txt> "
                     "<output_logits.txt> <summary.txt>\n";
        return 2;
    }

    try {
        std::ifstream cfg(argv[1]);
        if (!cfg) throw std::runtime_error("Failed to open config file");

        int input_dim = 0, hidden_dim = 0, output_dim = 0, num_steps = 0;
        int beta_num = 0, beta_den = 1, threshold = 1;
        cfg >> input_dim >> hidden_dim >> output_dim >> num_steps >> beta_num >> beta_den >> threshold;
        if (!cfg) throw std::runtime_error("Invalid config_fixed values");

        const auto w1 = read_memh_signed16(argv[2]);
        const auto w2 = read_memh_signed16(argv[3]);
        const auto spikes = read_memh_bit(argv[4]);
        const auto expected = read_dec_vector(argv[5]);

        if (static_cast<int>(w1.size()) != hidden_dim * input_dim) throw std::runtime_error("w1 size mismatch");
        if (static_cast<int>(w2.size()) != output_dim * hidden_dim) throw std::runtime_error("w2 size mismatch");
        if (static_cast<int>(spikes.size()) != num_steps * input_dim) throw std::runtime_error("spikes size mismatch");
        if (static_cast<int>(expected.size()) != output_dim) throw std::runtime_error("expected size mismatch");

        std::vector<int32_t> mem1(hidden_dim, 0), mem2(output_dim, 0), logits(output_dim, 0);
        std::vector<int32_t> spk1(hidden_dim, 0);

        for (int t = 0; t < num_steps; ++t) {
            for (int h = 0; h < hidden_dim; ++h) {
                int64_t cur1 = 0;
                for (int i = 0; i < input_dim; ++i) {
                    if (spikes[t * input_dim + i]) cur1 += w1[h * input_dim + i];
                }
                int32_t mem_pre = static_cast<int32_t>((static_cast<int64_t>(beta_num) * mem1[h]) / beta_den + cur1);
                if (mem_pre >= threshold) {
                    spk1[h] = 1;
                    mem1[h] = mem_pre - threshold;
                } else {
                    spk1[h] = 0;
                    mem1[h] = mem_pre;
                }
            }

            for (int o = 0; o < output_dim; ++o) {
                int64_t cur2 = 0;
                for (int h = 0; h < hidden_dim; ++h) {
                    if (spk1[h]) cur2 += w2[o * hidden_dim + h];
                }
                int32_t mem_pre = static_cast<int32_t>((static_cast<int64_t>(beta_num) * mem2[o]) / beta_den + cur2);
                if (mem_pre >= threshold) {
                    mem2[o] = mem_pre - threshold;
                    logits[o] += 1;
                } else {
                    mem2[o] = mem_pre;
                }
            }
        }

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

        std::cout << "Fixed reference complete. max_abs_err=" << max_abs_err << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}
