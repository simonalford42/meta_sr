"""Vectorised numpy port of the Johnston et al. (2014) Eisenia fetida energy-budget IBM,
exactly as used by van der Vaart et al. (2015) (NetLogo files g2002_1, ge2003_1, rv1990v_1,
rv1990c_1 from figshare 10.6084/m9.figshare.1420526, copied to
runs/pom-earthworm-figshare/netlogo_models/).

Simulates B parameter sets at once; each row of `theta` is one independent stochastic run.

Parameter order (same as the figshare reference table):
  B_0, E, E_c, E_f, E_s, h, IGm, M_b, M_c, M_m, M_p, r_B, r_m, s

NetLogo quirks reproduced on purpose (they are part of the published model):
  * func_response is a patch variable only refreshed in update-patch when food > 0 (stale after
    depletion / food addition);
  * food consumed is subtracted from the patch the worm is on *after* moving;
  * one cocoon at most per worm per day;
  * an adult demoted to juvenile by starvation also runs the juvenile routine in the same tick;
  * food removal (rv1990v, day 125): the .nlogo file lets patch food go negative (negative
    ingestion for a tick), but the published reference table (results_full_1e5) is only matched
    if food is clamped at 0, so we clamp (see validate_port.py);
  * growth is skipped entirely if it would take mass to >= M_m;
  * sum-hatchlings / mean-mass are over living worms only.
World: 2x2 wrapping patches over 0.0144 m^2 (patch side 0.06 m).
"""
import numpy as np

PNAMES = ["B_0", "E", "E_c", "E_f", "E_s", "h", "IGm", "M_b", "M_c", "M_m", "M_p", "r_B", "r_m", "s"]
LIT = np.array([967, 0.25, 7.0, 10.6, 3.6, 3.5, 0.70, 0.011, 0.015, 0.5, 0.25, 0.177, 0.182, 0.004])
PRIOR_SDLOG = 0.3536
BOLTZ = 8.62e-5
TREF = 298.15
SCAPE = 0.0144
NPATCH = 4
HET_CV = 0.0   # >0: per-worm lognormal multipliers on IGm and r_B (individual variation; NOT in the
               # published model -- only used to generate misspecified synthetic "truth")


def sample_prior(n, rng):
    return LIT * np.exp(rng.normal(0.0, PRIOR_SDLOG, size=(n, len(LIT))))


class Sim:
    def __init__(self, theta, n_juv, total_food, temp_c, max_worms, rng):
        th = np.atleast_2d(np.asarray(theta, float))
        B = th.shape[0]
        self.B, self.W, self.rng = B, max_worms, rng
        c = lambda i: th[:, i:i + 1]
        (self.B0, self.E, self.Ec, self.Ef, self.Es, self.h, self.IGm, self.Mb, self.Mc,
         self.Mm, self.Mp, self.rB, self.rm, self.s) = [c(i) for i in range(14)]
        if HET_CV > 0:
            self.IGm = self.IGm * np.exp(rng.normal(0, HET_CV, (B, max_worms)))
            self.rB = self.rB * np.exp(rng.normal(0, HET_CV, (B, max_worms)))
        T = 273.15 + temp_c
        self.arr = np.exp((-self.E / BOLTZ) * (1.0 / T - 1.0 / TREF))
        self.bmr_c = self.B0 * np.exp(-self.E / (BOLTZ * T))
        self.ecs = self.Ec + self.Es
        self.food_per_patch = total_food / NPATCH
        self.half = (self.h * SCAPE / 0.01) / NPATCH            # (B,1)
        self.step_len = self.s / (np.sqrt(SCAPE) / 2.0)          # patches per day
        shp = (B, max_worms)
        self.alive = np.zeros(shp, bool)
        self.alive[:, :n_juv] = True
        self.adult = np.zeros(shp, bool)
        self.mass = np.where(self.alive, self.Mb, 0.0)
        self.resmax = self.mass / 2 * self.Ec
        self.res = self.resmax.copy()
        self.R = np.zeros(shp)
        self.hatch = np.zeros(shp)
        self.ing = np.zeros(shp)
        self.el = np.zeros(shp)
        self.x = rng.uniform(0, 2, shp)
        self.y = rng.uniform(0, 2, shp)
        self.head = rng.integers(0, 360, shp).astype(float)
        self.food = np.full((B, NPATCH), float(self.food_per_patch))
        self.fr = np.zeros((B, NPATCH))
        self.t = 0
        self.update_patch()

    # ------------------------------------------------------------------ helpers
    def patch(self):
        return (np.floor(self.x).astype(int) % 2) + 2 * (np.floor(self.y).astype(int) % 2)

    def randomise_positions(self):
        self.x = self.rng.uniform(0, 2, self.x.shape)
        self.y = self.rng.uniform(0, 2, self.y.shape)

    def mean_mass(self):
        n = self.alive.sum(1)
        s = np.where(self.alive, self.mass, 0).sum(1)
        return np.where(n > 0, s / np.maximum(n, 1), 0.0)

    def sum_hatch(self):
        return np.where(self.alive, self.hatch, 0).sum(1)

    # ------------------------------------------------------------------ processes
    def calc_ingestion(self):
        p = self.patch()
        fr = np.take_along_axis(self.fr, p, 1)
        ing = self.IGm * self.arr * fr * np.where(self.alive, self.mass, 0) ** (2 / 3)
        ing = np.where(self.alive, ing, 0.0)
        for k in range(NPATCH):
            here = (p == k) & self.alive
            f = self.food[:, k:k + 1]
            ing = np.where(here & (f == 0), 0.0, ing)
            tot = np.where(here, ing, 0).sum(1, keepdims=True)
            cnt = here.sum(1, keepdims=True)
            over = (tot > f) & (cnt > 0)
            ing = np.where(here & over, f / np.maximum(cnt, 1), ing)
            self.food[:, k] = np.where(over[:, 0], 0.0, self.food[:, k])
        self.ing = ing

    def maintenance(self, m):
        bmr = self.bmr_c * np.maximum(self.mass, 0) ** 0.75
        ok = m & (self.el > bmr)
        short = m & ~(self.el > bmr)
        self.el = np.where(ok, self.el - bmr, self.el)
        self.res = np.where(short, self.res - (bmr - self.el), self.res)
        self.el = np.where(short, 0.0, self.el)
        starve = m & (self.res < 0.5 * self.resmax)
        self.mass = np.where(starve, self.mass - bmr / self.ecs, self.mass)
        self.res = np.where(starve, self.res + bmr, self.res)
        demote = starve & self.adult & (self.mass < self.Mp)
        self.adult = np.where(demote, False, self.adult)
        die = starve & (self.mass < self.Mb)
        self.alive = np.where(die, False, self.alive)
        return demote

    def reproduction(self, m):
        maxR = self.rm * self.arr * self.mass
        eR = np.minimum(self.el, maxR)
        self.el = np.where(m, self.el - eR, self.el)
        top = m & (eR < maxR) & (self.res > 0.5 * self.resmax)
        self.res = np.where(top, self.res - (maxR - eR), self.res)
        eR = np.where(top, maxR, eR)
        self.R = np.where(m, self.R + eR, self.R)
        cost = self.Mc * self.ecs
        rep = m & (self.R >= cost)
        self.hatch = self.hatch + rep
        self.R = np.where(rep, self.R - cost, self.R)

    def growth(self, m):
        maxG = self.rB * self.arr * (self.Mm ** (1 / 3) * np.maximum(self.mass, 0) ** (2 / 3) - self.mass)
        eG = np.minimum(maxG * self.ecs, self.el)
        to_grow = np.where(maxG > 0, eG / self.ecs, 0.0)
        g = m & (self.mass + to_grow < self.Mm)
        self.mass = np.where(g, self.mass + to_grow, self.mass)
        self.el = np.where(g, self.el - eG, self.el)

    def reserves(self, m):
        self.resmax = np.where(m, self.mass / 2 * self.Ec, self.resmax)
        add = m & (self.el > 0)
        self.res = np.where(add, self.res + self.el * self.Ec / self.ecs, self.res)
        self.res = np.where(m, np.minimum(self.res, self.resmax), self.res)

    def move(self, m):
        turn = self.rng.integers(0, 90, self.head.shape) - self.rng.integers(0, 90, self.head.shape)
        self.head = np.where(m, self.head + turn, self.head)
        rad = np.deg2rad(self.head)
        self.x = np.where(m, (self.x + self.step_len * np.sin(rad)) % 2, self.x)
        self.y = np.where(m, (self.y + self.step_len * np.cos(rad)) % 2, self.y)

    def update_patch(self):
        p = self.patch()
        for k in range(NPATCH):
            here = (p == k) & self.alive
            tot = np.where(here, self.ing, 0).sum(1)
            f = self.food[:, k]
            pos = f > 0
            nf = np.maximum(f - tot, 0.0)
            self.food[:, k] = np.where(pos, nf, f)
            self.fr[:, k] = np.where(pos, nf / (nf + self.half[:, 0]), self.fr[:, k])

    def go(self, experiment_hook):
        experiment_hook(self)
        self.calc_ingestion()
        A = self.alive & self.adult
        self.el = np.where(A, self.ing * self.Ef, self.el)
        demoted = self.maintenance(A)
        A = A & self.alive
        self.reproduction(A)
        self.growth(A)
        self.reserves(A)
        self.move(A)
        J = self.alive & (~self.adult)            # includes adults demoted this tick
        self.el = np.where(J, self.ing * self.Ef, self.el)
        self.maintenance(J)
        J = J & self.alive
        self.growth(J)
        self.adult = np.where(J & (self.mass >= self.Mp), True, self.adult)
        self.reserves(J)
        self.move(J)
        self.update_patch()
        self.t += 1

    # ------------------------------------------------------------------ cohort ops
    def kill_random(self, n):
        for b in range(self.B):
            idx = np.flatnonzero(self.alive[b])
            if len(idx) >= n:
                self.alive[b, self.rng.choice(idx, n, replace=False)] = False

    def add_adults(self, n):
        for _ in range(n):
            free = np.argmax(~self.alive, axis=1)      # first free slot (W sized to fit)
            b = np.arange(self.B)
            nalive = self.alive.sum(1)
            s = np.where(self.alive, self.mass, 0).sum(1)
            newm = s / (nalive + 1)                       # NetLogo: own mass (0) is in the mean
            ok = nalive > 0
            pk = self.rng.integers(0, NPATCH, self.B)
            sl = free[ok]; bb = b[ok]
            self.alive[bb, sl] = True
            self.adult[bb, sl] = True
            self.mass[bb, sl] = newm[ok]
            self.resmax[bb, sl] = newm[ok] / 2 * self.Ec[bb, 0]
            self.res[bb, sl] = self.resmax[bb, sl]
            self.R[bb, sl] = 0; self.hatch[bb, sl] = 0; self.ing[bb, sl] = 0; self.el[bb, sl] = 0
            self.x[bb, sl] = (pk[ok] % 2) + 0.5
            self.y[bb, sl] = (pk[ok] // 2) + 0.5
            self.head[bb, sl] = self.rng.integers(0, 360, ok.sum())


# ---------------------------------------------------------------------- experiments
def run_g2002(theta, rng, n_juv=5, total_food=150.0, temp=20.0, days=182):
    sim = Sim(theta, n_juv, total_food, temp, n_juv, rng)

    def hook(s):
        if s.t % 7 == 0:
            s.randomise_positions()
    out = [sim.mean_mass()]
    while sim.t < days:
        sim.go(hook)
        if sim.t % 7 == 0:
            out.append(sim.mean_mass())
    return np.stack(out, 1)


def run_ge2003(theta, rng):
    sim = Sim(theta, 8, 100.0, 20.0, 12, rng)

    def hook(s):
        if s.t % 7 == 0:
            s.randomise_positions()
        if s.t == 80:
            s.kill_random(2)
        if s.t == 161:
            s.add_adults(4)
        if s.t in (161, 315):
            s.hatch[:] = 0
            s.food += s.food_per_patch
    out = [sim.mean_mass()]
    while sim.t < 420:
        sim.go(hook)
        if sim.t % 7 == 0:
            out.append(sim.mean_mass())
    return np.stack(out, 1)


def _rv_hook_factory(variable, temp_food_scale=1.0):
    def hook(s):
        if s.t >= 25 and s.t % 10 == 5:
            s.randomise_positions()
            s.hatch[:] = 0
        f = temp_food_scale / NPATCH
        if variable:
            if s.t == 25: s.food[:] = 10 * f
            if s.t == 35: s.food += 10 * f
            if s.t == 85: s.food += 50 * f
            if s.t == 125: s.food = np.maximum(s.food - 25 * f, 0.0)  # clamp: matches NetLogo reference table
            if s.t == 165: s.food += 25 * f
        else:
            if s.t == 25: s.food[:] = 20 * f
            if s.t > 25 and (s.t - 25) % 20 == 0: s.food += 20 * f
    return hook


def run_rv1990(theta, rng, variable, n_juv=10, temp=25.0, food_scale=1.0):
    """Returns (B, 36): interleaved (cocoons, mass) at t=25,35..185 then cocoons at 195, 205."""
    sim = Sim(theta, n_juv, 40.0, temp, n_juv, rng)
    hook = _rv_hook_factory(variable, food_scale)
    while sim.t < 25:
        sim.go(hook)
    out = [sim.sum_hatch(), sim.mean_mass()]
    for _ in range(16):
        for _ in range(10):
            sim.go(hook)
        out += [sim.sum_hatch(), sim.mean_mass()]
    for _ in range(2):
        for _ in range(10):
            sim.go(hook)
        out.append(sim.sum_hatch())
    return np.stack(out, 1)


def run_all(theta, rng):
    """160 summary statistics in the order of the figshare all_data.txt."""
    return np.concatenate([run_g2002(theta, rng), run_ge2003(theta, rng),
                           run_rv1990(theta, rng, True), run_rv1990(theta, rng, False)], 1)


# column index sets (0-based) into the 160-vector
COLS = {
    "g02": np.arange(0, 27),
    "ge03": np.arange(27, 88),
    "rv90v_m": np.arange(89, 122, 2),
    "rv90v_c": np.r_[np.arange(88, 123, 2), 123],
    "rv90c_m": np.arange(125, 158, 2),
    "rv90c_c": np.r_[np.arange(124, 159, 2), 159],
}
EXP_COLS = {"exp1": COLS["g02"], "exp2": COLS["ge03"],
            "exp3": np.r_[88:124], "exp4": np.r_[124:160]}
MASS_COLS = np.r_[COLS["g02"], COLS["ge03"], COLS["rv90v_m"], COLS["rv90c_m"]]
COC_COLS = np.r_[COLS["rv90v_c"], COLS["rv90c_c"]]
