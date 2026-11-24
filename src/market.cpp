#include "qubo/market.hpp"

#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "qubo/rng.hpp"

namespace qubo {

Market simulate_market(const MarketConfig& cfg) {
    if (cfg.n_assets < 2 || cfg.n_days < 2 || cfg.n_sectors < 1) throw std::invalid_argument("market too small");
    if (cfg.garch_alpha + cfg.garch_beta >= 1.0) throw std::invalid_argument("GARCH must be stationary");
    if (cfg.tail_nu <= 2.0) throw std::invalid_argument("t innovations need nu > 2 for finite variance");

    Rng rng(cfg.seed);
    const int n = cfg.n_assets;
    const double day = 1.0 / 252.0;
    const double sd_day = std::sqrt(day);

    Market m;
    m.returns = Matrix(static_cast<std::size_t>(cfg.n_days), static_cast<std::size_t>(n));
    std::vector<double> beta(n), gamma(n), idio(n), alpha(n);
    for (int i = 0; i < n; ++i) {
        m.sector.push_back(i % cfg.n_sectors);
        m.names.push_back("S" + std::to_string(i % cfg.n_sectors) + "_A" + std::to_string(i));
        beta[i] = 0.6 + 0.8 * rng.uniform();
        gamma[i] = 0.5 + 0.5 * rng.uniform();
        idio[i] = cfg.idio_vol_min + (cfg.idio_vol_max - cfg.idio_vol_min) * rng.uniform();
        alpha[i] = cfg.alpha_dispersion * rng.normal();  // start from the stationary law
    }

    // GARCH(1,1) on the daily market factor, started at its unconditional
    // variance: omega / (1 - a - b) = market_vol^2 / 252.
    const double var_lr = cfg.market_vol * cfg.market_vol * day;
    const double omega = var_lr * (1.0 - cfg.garch_alpha - cfg.garch_beta);
    double var_t = var_lr;
    double last_mkt = 0.0;

    // AR(1) for alpha with the requested half-life, variance preserved.
    const double phi = std::exp(-std::log(2.0) / cfg.alpha_half_life);
    const double innov = std::sqrt(1.0 - phi * phi) * cfg.alpha_dispersion;

    std::vector<double> sector_f(cfg.n_sectors);
    for (int t = 0; t < cfg.n_days; ++t) {
        var_t = omega + cfg.garch_alpha * last_mkt * last_mkt + cfg.garch_beta * var_t;
        const double mkt = std::sqrt(var_t) * rng.student_t_unit(cfg.tail_nu);
        last_mkt = mkt;
        for (auto& f : sector_f) f = cfg.sector_vol * sd_day * rng.student_t_unit(cfg.tail_nu);
        for (int i = 0; i < n; ++i) {
            alpha[i] = phi * alpha[i] + innov * rng.normal();
            const double eps = idio[i] * sd_day * rng.student_t_unit(cfg.tail_nu);
            // Log-return drift includes the usual -sigma^2/2 so that the
            // arithmetic mean return matches mean_return + alpha.
            const double total_var = beta[i] * beta[i] * var_lr + gamma[i] * gamma[i] * cfg.sector_vol * cfg.sector_vol * day +
                                     idio[i] * idio[i] * day;
            const double drift = (cfg.mean_return + alpha[i]) * day - 0.5 * total_var;
            m.returns(t, i) = drift + beta[i] * mkt + gamma[i] * sector_f[m.sector[i]] + eps;
        }
    }
    return m;
}

namespace {
std::vector<std::string> split(const std::string& line) {
    std::vector<std::string> out;
    std::stringstream ss(line);
    std::string cell;
    while (std::getline(ss, cell, ',')) {
        while (!cell.empty() && (cell.back() == '\r' || cell.back() == ' ')) cell.pop_back();
        out.push_back(cell);
    }
    return out;
}
}  // namespace

Market load_csv(const std::string& path, bool prices) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open " + path);
    std::string line;
    if (!std::getline(in, line)) throw std::runtime_error("empty file " + path);
    Market m;
    m.names = split(line);
    const std::size_t n = m.names.size();
    std::vector<std::vector<double>> rows;
    std::size_t line_no = 1;
    while (std::getline(in, line)) {
        ++line_no;
        if (line.empty()) continue;
        auto cells = split(line);
        if (cells.size() != n) {
            throw std::runtime_error(path + ":" + std::to_string(line_no) + ": expected " + std::to_string(n) + " columns");
        }
        std::vector<double> r(n);
        for (std::size_t j = 0; j < n; ++j) {
            try {
                r[j] = std::stod(cells[j]);
            } catch (const std::exception&) {
                throw std::runtime_error(path + ":" + std::to_string(line_no) + ": not a number '" + cells[j] + "'");
            }
            if (prices && r[j] <= 0.0) throw std::runtime_error(path + ":" + std::to_string(line_no) + ": non-positive price");
        }
        rows.push_back(std::move(r));
    }
    // Checked before the subtraction below, which would wrap around on an
    // empty file.
    if (rows.size() < 3) throw std::runtime_error("not enough rows in " + path);
    const std::size_t t_len = prices ? rows.size() - 1 : rows.size();
    m.returns = Matrix(t_len, n);
    for (std::size_t t = 0; t < t_len; ++t) {
        for (std::size_t j = 0; j < n; ++j) {
            m.returns(t, j) = prices ? std::log(rows[t + 1][j] / rows[t][j]) : rows[t][j];
        }
    }
    m.sector.assign(n, -1);
    return m;
}

void save_returns_csv(const Market& m, const std::string& path) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("cannot write " + path);
    for (std::size_t j = 0; j < m.names.size(); ++j) out << (j ? "," : "") << m.names[j];
    out << "\n";
    out.precision(10);
    for (std::size_t t = 0; t < m.returns.rows(); ++t) {
        for (std::size_t j = 0; j < m.returns.cols(); ++j) out << (j ? "," : "") << m.returns(t, j);
        out << "\n";
    }
}

}  // namespace qubo

// -- laklaa.com
