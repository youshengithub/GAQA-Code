"""GAQA: query-efficient generation of adversarial examples for defensive DNNs.

This module implements Algorithm 1 (the GA search) and Algorithm 2 (PSO-Jump)
of the paper, with the operators named in Section IV-E:

* perturbation-varied reproduction   (Section IV-E1, alpha ladder along Delta)
* block-based crossover              (Section IV-E2, Fig. 3)
* shuffle-based crossover            (Section IV-E3, Eq. 9)
* discrete mutation                  (Section IV-E4, Eq. 10-11)
* adaptation of the search parameters(Section IV-F, Eq. 12-15)

A perturbation solution is the matrix chromosome ``delta`` of Section IV-B: one
value per pixel, bounded by ``[-eps, eps]``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from fitness import fitness, mcsd_reference

if TYPE_CHECKING:  # the search itself only needs a probabilities() endpoint
    from victim import Victim

ABLATIONS = ("init", "selection", "pso", "var", "block", "shuffle", "mutation")


@dataclass
class Result:
    """Outcome of one GAQA run on one image."""

    delta: np.ndarray
    fitness: float
    prediction: int
    success: bool
    queries: int
    queries_to_success: int
    iterations: int
    success_iteration: int | None
    history: list[float] = field(default_factory=list)


class Chromosome:
    """A perturbation solution together with its fitness value (Section IV-B)."""

    __slots__ = ("data", "score", "prediction")

    def __init__(self, data: np.ndarray) -> None:
        self.data = data
        self.score = 0.0
        self.prediction = -1

    def copy(self) -> "Chromosome":
        clone = Chromosome(self.data.copy())
        clone.score = self.score
        clone.prediction = self.prediction
        return clone


class GAQA:
    """Elites-preferred GA with PSO-Jump, as described in Algorithm 1."""

    def __init__(
        self,
        victim: "Victim",
        image: np.ndarray,
        label: int,
        eps: float = 8 / 255,
        population: int = 100,
        rho: int = 10,
        theta: int = 1000,
        iters: int = 1000,
        nu: int = 300,
        psi: float = 0.1,
        pso_iters: int = 20,
        pso_particles: int = 10,
        c1: float = 0.5,
        c2: float = 0.5,
        selection_pi: int = 8,
        alpha_step: float = 0.2,
        seed: int | None = None,
        ablate: tuple[str, ...] = (),
        verbose: bool = False,
    ) -> None:
        unknown = set(ablate) - set(ABLATIONS)
        if unknown:
            raise ValueError(f"unknown ablation(s): {sorted(unknown)}")
        self.victim = victim
        self.image = image
        self.label = int(label)
        self.eps = float(eps)
        self.shape = image.shape
        self.population_size = int(population)
        self.rho = int(rho)
        self.theta = float(theta)
        self.iters = int(iters)
        self.nu = int(nu)
        self.psi = float(psi)
        self.pso_iters = int(pso_iters)
        self.pso_particles = int(pso_particles)
        self.c1 = float(c1)
        self.c2 = float(c2)
        self.selection_pi = float(selection_pi)
        self.ablate = tuple(ablate)
        self.verbose = verbose

        self.rng = np.random.default_rng(seed)
        self.mcsd_max = mcsd_reference(self.shape, self.eps)

        # Algorithm 1, lines 2-3: jumping threshold and reproduction parameters.
        self.alpha = [alpha_step * k for k in range(1, self.rho + 1)]
        self.gamma = 3
        self.eta = min(10, self.population_size)
        self.prob_mutation = 0.02
        self.prob_shuffle = 0.02

        self.population: list[Chromosome] = []
        self.last_best = Chromosome(np.zeros(self.shape, dtype=np.float32))
        self.queries_start = victim.queries

        self._h_divisors = [i + 1 for i in range(self.shape[1]) if self.shape[1] % (i + 1) == 0]
        self._w_divisors = [i + 1 for i in range(self.shape[2]) if self.shape[2] % (i + 1) == 0]

    # ------------------------------------------------------------------ setup
    def _random_perturbation(self) -> np.ndarray:
        return self.rng.uniform(-self.eps, self.eps, size=self.shape).astype(np.float32)

    def _init_population(self) -> None:
        """Section IV-C.  ``init`` ablation starts from the zero perturbation."""
        if "init" in self.ablate:
            self.population = [Chromosome(np.zeros(self.shape, dtype=np.float32))
                               for _ in range(self.population_size)]
        else:
            self.population = [Chromosome(self._random_perturbation())
                               for _ in range(self.population_size)]

    # --------------------------------------------------------------- queries
    def _evaluate(self, population: list[Chromosome]) -> np.ndarray:
        """Query the victim for the whole population and score it (Eq. 6)."""
        deltas = np.stack([c.data for c in population])
        probabilities = self.victim.probabilities(self.image, deltas)
        scores = fitness(probabilities, self.label, deltas, self.eps, self.psi, self.mcsd_max)
        predictions = probabilities.argmax(axis=1)
        for chromo, score, prediction in zip(population, scores, predictions):
            chromo.score = float(score)
            chromo.prediction = int(prediction)
        return probabilities

    # ------------------------------------------------------------- selection
    def _select(self, population: list[Chromosome]) -> list[Chromosome]:
        """Elites-preferred probabilistic selection, Eq. 8.

        ``Omega(i) = (n / (n + i - 1)) ** pi`` for the i-th solution of the
        population sorted by decreasing fitness; the elite always survives.
        """
        if "selection" in self.ablate:
            return list(population)
        size = len(population)
        kept = [population[0]]
        for index, chromo in enumerate(population[1:], start=2):
            if self.rng.random() <= (size / (size + index - 1)) ** self.selection_pi:
                kept.append(chromo)
        return kept

    # ------------------------------------------------------------ adaptation
    def _predicted_steps(self, history: list[float], iteration: int) -> float:
        """Eq. 12: predicted number of remaining iterations."""
        if iteration < 100 or len(history) <= 100:
            return 0.0
        denominator = history[-1] - history[-101]
        numerator = 100.0 * (1.0 + 2.0 * self.psi - history[-1])
        if denominator <= 1e-9:
            return 1e4  # the search has stalled: saturate the prediction
        return float(min(max(numerator / denominator, 0.0), 1e4))

    def _adapt(self, beta: float) -> None:
        """Eq. 13-15: update gamma, eta and the mutation/shuffling probabilities."""
        self.gamma = max(self.gamma, 3 + 2 * int(beta // 250))
        self.eta = int(np.clip(10 - (beta - 2500) / 1000, 5, min(10, self.population_size)))
        probability = float(np.clip((beta - 5000) / 2e5, 0.02, 0.1))
        self.prob_mutation = probability
        self.prob_shuffle = probability

    # ----------------------------------------------------------- reproduction
    def _perturbation_varied(self, population: list[Chromosome]) -> list[Chromosome]:
        """Section IV-E1: children along ``Delta = P_1^i - P_1^(i-1)``."""
        if "var" in self.ablate:
            return []
        elite = population[0]
        direction = elite.data - self.last_best.data
        children = []
        for alpha in self.alpha:
            data = np.clip(elite.data + alpha * direction, -self.eps, self.eps)
            children.append(Chromosome(data.astype(np.float32)))
        return children

    def _block_crossover(self, population: list[Chromosome]) -> list[Chromosome]:
        """Section IV-E2 / Fig. 3: two children per pair of the top ``eta`` fathers."""
        if "block" in self.ablate:
            return []
        fathers = population[:min(int(self.eta), len(population))]
        children = []
        for j in range(len(fathers) - 1):
            for k in range(j + 1, len(fathers)):
                first, second = self._crossover_pair(fathers[j], fathers[k])
                children.append(first)
                children.append(second)
        return children

    def _crossover_pair(self, a: Chromosome, b: Chromosome) -> tuple[Chromosome, Chromosome]:
        child_a = Chromosome(np.empty(self.shape, dtype=np.float32))
        child_b = Chromosome(np.empty(self.shape, dtype=np.float32))
        for channel in range(self.shape[0]):
            h_cut = int(self.rng.choice(self._h_divisors))
            w_cut = int(self.rng.choice(self._w_divisors))
            h_block = self.shape[1] // h_cut
            w_block = self.shape[2] // w_cut
            flip = False
            for i in range(h_cut):
                for j in range(w_cut):
                    rows = slice(i * h_block, (i + 1) * h_block)
                    cols = slice(j * w_block, (j + 1) * w_block)
                    source_a, source_b = (b, a) if flip else (a, b)
                    child_a.data[channel, rows, cols] = source_a.data[channel, rows, cols]
                    child_b.data[channel, rows, cols] = source_b.data[channel, rows, cols]
                    flip = not flip
        return child_a, child_b

    def _shuffle_crossover(self, population: list[Chromosome]) -> list[Chromosome]:
        """Section IV-E3 / Eq. 9: elites shuffled gene-wise with the inferiors."""
        if "shuffle" in self.ablate:
            return []
        half = len(population) // 2
        children = []
        for j in range(half):
            elite = population[j]
            inferior = population[half + j]
            mask = self.rng.random(self.shape) < self.prob_shuffle
            children.append(Chromosome(np.where(mask, elite.data, inferior.data)
                                       .astype(np.float32)))
        return children

    def _mutate(self, chromo: Chromosome) -> None:
        """Section IV-E4 / Eq. 10-11: discrete, probabilistic mutation in place."""
        if "mutation" in self.ablate:
            return
        levels = max(int(self.gamma), 3)
        indices = self.rng.integers(0, levels, size=self.shape)
        values = (2.0 * indices / (levels - 1) - 1.0) * self.eps
        mask = self.rng.random(self.shape) < self.prob_mutation
        chromo.data = np.where(mask, values, chromo.data).astype(np.float32)

    # -------------------------------------------------------------- PSO-Jump
    def _pso_jump(self, population: list[Chromosome]) -> list[Chromosome]:
        """Algorithm 2: escape the local optimum without dropping the population."""
        particles = [c.copy() for c in population[:self.pso_particles]]
        if not particles:
            return population
        velocities = [np.zeros(self.shape, dtype=np.float32) for _ in particles]
        personal_best = [p.copy() for p in particles]
        global_best = particles[0].copy()
        if self.verbose:
            print(f"    PSO-Jump over {len(particles)} particles x {self.pso_iters} steps")
        for _ in range(self.pso_iters):
            for index, particle in enumerate(particles):
                to_global = global_best.data - particle.data
                to_personal = personal_best[index].data - particle.data
                velocities[index] = (velocities[index]
                                     + self.c1 * self.rng.random() * to_global
                                     + self.c2 * self.rng.random() * to_personal)
                particle.data = np.clip(particle.data + velocities[index],
                                        -self.eps, self.eps).astype(np.float32)
            probabilities = self._evaluate(particles)
            if (probabilities.argmax(axis=1) != self.label).any():
                break  # an effective AE has been generated
            for index, particle in enumerate(particles):
                if particle.score > personal_best[index].score:
                    personal_best[index] = particle.copy()
                if particle.score > global_best.score:
                    global_best = particle.copy()
        particles.sort(key=lambda c: c.score, reverse=True)
        return particles

    # ------------------------------------------------------------------- loop
    def _trim(self, selected: list[Chromosome], children: list[Chromosome]) -> list[Chromosome]:
        """Fill the next generation up to ``Q`` solutions, elites first."""
        room = self.population_size - len(selected)
        if room <= 0:
            return selected[:self.population_size]
        if len(children) > room:
            picks = self.rng.choice(len(children), size=room, replace=False)
            children = [children[i] for i in picks]
        return selected + children

    def run(self) -> Result:
        """Algorithm 1."""
        self._init_population()
        self.queries_start = self.victim.queries

        deadline = self.iters
        found = False
        success_iteration: int | None = None
        queries_to_success = 0
        history: list[float] = []
        iteration = 0

        while iteration < deadline:
            iteration += 1
            self._evaluate(self.population)
            self.population.sort(key=lambda c: c.score, reverse=True)

            best = self.population[0]
            # Lines 10-11: an effective AE extends the budget by nu iterations
            # so that the evasiveness objective gets optimised as well.  The
            # paper tests Phi > 1 here, but with Eq. 6 that is not equivalent to
            # a misclassification (see the notes in the README), so the label
            # returned by the victim is used instead -- it is what ASR measures.
            if not found and best.prediction != self.label:
                deadline = iteration + self.nu
                found = True
                success_iteration = iteration
                queries_to_success = self.victim.queries - self.queries_start
            history.append(best.score)

            if self.verbose:
                print(f"  it {iteration:4d}/{deadline:4d}  best={best.score:.4f}  "
                      f"q={self.victim.queries}  gamma={self.gamma}  eta={self.eta}")

            selected = self._select(self.population)

            beta = self._predicted_steps(history, iteration)
            if beta > self.theta and "pso" not in self.ablate:
                selected = self._pso_jump(selected)

            # The reproduction operators draw their fathers from the whole
            # sorted generation (Section IV-E "elite population and inferior
            # population"), not from the survivors of the selection step.
            children = (self._perturbation_varied(self.population)
                        + self._block_crossover(self.population)
                        + self._shuffle_crossover(self.population))
            for child in children:
                self._mutate(child)

            self.last_best = self.population[0].copy()
            self.population = self._trim(selected, children)
            self._adapt(beta)

        self.population.sort(key=lambda c: c.score, reverse=True)
        best = self.population[0]
        prediction = int(self.victim.probabilities(self.image, best.data[None]).argmax(axis=1)[0])
        return Result(
            delta=best.data,
            fitness=best.score,
            prediction=prediction,
            success=prediction != self.label,
            queries=self.victim.queries - self.queries_start,
            queries_to_success=queries_to_success or self.victim.queries - self.queries_start,
            iterations=iteration,
            success_iteration=success_iteration,
            history=history,
        )
