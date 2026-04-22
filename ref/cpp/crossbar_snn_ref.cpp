#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::vector<double> read_vector_file(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Failed to open: " + path);
    }
    std::vector<double> values;
    double v = 0.0;
    while (in >> v) {
        values.push_back(v);
    }
    return values;
}

void require_size(const std::vector<double>& values, size_t expected, const std::string& name) {
    if (values.size() != expected) {
        std::ostringstream oss;
        oss << "Invalid " << name << " size. Got " << values.size() << ", expected " << expected;
        throw std::runtime_error(oss.str());
    }
}

double dot_row(const std::vector<double>& mat, size_t row, size_t cols, const std::vector<double>& vec) {
    const size_t base = row * cols;
    double acc = 0.0;
    for (size_t i = 0; i < cols; ++i) {
        acc += mat[base + i] * vec[i];
    }
    return acc;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 7) {
        std::cerr << "Usage: " << argv[0]
                  << " <config.txt> <w1.txt> <w2.txt> <spikes.txt> <expected_logits.txt> <output_logits.txt>\n";
        return 2;
    }

    const std::string config_path = argv[1];
    const std::string w1_path = argv[2];
    const std::string w2_path = argv[3];
    const std::string spikes_path = argv[4];
    const std::string expected_path = argv[5];
    const std::string output_path = argv[6];

    try {
        std::ifstream cfg(config_path);
        if (!cfg) {
            throw std::runtime_error("Failed to open config file: " + config_path);
        }

        int input_dim = 0;
        int hidden_dim = 0;
        int output_dim = 0;
        int num_steps = 0;
        double beta = 0.0;
        double threshold = 1.0;
        std::string mode = "discrete";
        cfg >> input_dim >> hidden_dim >> output_dim >> num_steps >> beta >> threshold;
        if (!(cfg >> mode)) {
            mode = "discrete";
        }
        if (!cfg || input_dim <= 0 || hidden_dim <= 0 || output_dim <= 0 || num_steps <= 0) {
            throw std::runtime_error("Invalid configuration values.");
        }
        if (mode != "discrete" && mode != "snntorch") {
            throw std::runtime_error("Invalid mode in config. Expected 'discrete' or 'snntorch'.");
        }

        const std::vector<double> w1 = read_vector_file(w1_path);
        const std::vector<double> w2 = read_vector_file(w2_path);
        const std::vector<double> spikes = read_vector_file(spikes_path);
        const std::vector<double> expected = read_vector_file(expected_path);

        require_size(w1, static_cast<size_t>(hidden_dim) * static_cast<size_t>(input_dim), "w1");
        require_size(w2, static_cast<size_t>(output_dim) * static_cast<size_t>(hidden_dim), "w2");
        require_size(spikes, static_cast<size_t>(num_steps) * static_cast<size_t>(input_dim), "spikes");
        require_size(expected, static_cast<size_t>(output_dim), "expected");

        std::vector<double> mem1(hidden_dim, 0.0);
        std::vector<double> mem2(output_dim, 0.0);
        std::vector<double> logits(output_dim, 0.0);
        std::vector<double> spk1(hidden_dim, 0.0);
        std::vector<double> spk2(output_dim, 0.0);

        for (int t = 0; t < num_steps; ++t) {
            const size_t spike_base = static_cast<size_t>(t) * static_cast<size_t>(input_dim);
            std::vector<double> spk_in(input_dim, 0.0);
            for (int i = 0; i < input_dim; ++i) {
                spk_in[i] = spikes[spike_base + static_cast<size_t>(i)];
            }

            for (int h = 0; h < hidden_dim; ++h) {
                const size_t hs = static_cast<size_t>(h);
                const double cur1 = dot_row(w1, hs, static_cast<size_t>(input_dim), spk_in);
                if (mode == "snntorch") {
                    const double reset = mem1[hs] > threshold ? 1.0 : 0.0;
                    const double mem_new = beta * mem1[hs] + cur1 - reset * threshold;
                    const double spike = mem_new > threshold ? 1.0 : 0.0;
                    mem1[hs] = mem_new;
                    spk1[hs] = spike;
                } else {
                    const double mem_pre = beta * mem1[hs] + cur1;
                    const double spike = mem_pre >= threshold ? 1.0 : 0.0;
                    mem1[hs] = mem_pre - spike * threshold;
                    spk1[hs] = spike;
                }
            }

            for (int o = 0; o < output_dim; ++o) {
                const size_t os = static_cast<size_t>(o);
                const double cur2 = dot_row(w2, os, static_cast<size_t>(hidden_dim), spk1);
                if (mode == "snntorch") {
                    const double reset = mem2[os] > threshold ? 1.0 : 0.0;
                    const double mem_new = beta * mem2[os] + cur2 - reset * threshold;
                    const double spike = mem_new > threshold ? 1.0 : 0.0;
                    mem2[os] = mem_new;
                    spk2[os] = spike;
                    logits[os] += spike;
                } else {
                    const double mem_pre = beta * mem2[os] + cur2;
                    const double spike = mem_pre >= threshold ? 1.0 : 0.0;
                    mem2[os] = mem_pre - spike * threshold;
                    spk2[os] = spike;
                    logits[os] += spike;
                }
            }
        }

        double max_abs_err = 0.0;
        double mse = 0.0;
        for (int o = 0; o < output_dim; ++o) {
            const double err = std::fabs(logits[static_cast<size_t>(o)] - expected[static_cast<size_t>(o)]);
            max_abs_err = std::max(max_abs_err, err);
            mse += err * err;
        }
        mse /= static_cast<double>(output_dim);

        std::ofstream out(output_path);
        if (!out) {
            throw std::runtime_error("Failed to open output file: " + output_path);
        }
        out << std::setprecision(10);
        for (int o = 0; o < output_dim; ++o) {
            out << logits[static_cast<size_t>(o)] << "\n";
        }

        std::cout << "Reference inference complete.\n";
        std::cout << "max_abs_err=" << max_abs_err << "\n";
        std::cout << "mse=" << mse << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}
