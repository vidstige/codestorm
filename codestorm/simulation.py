import numpy as np


def constant(stiffness, age):
    return stiffness


def replace_index(spring, a, b):
    i, j, length, stiffness, timestamp = spring
    if i == a:
        i = b
    if j == a:
        j = b
    return i, j, length, stiffness, timestamp


class Simulation:
    def __init__(self, stiffness=constant):
        self.t = 0
        self.state = np.empty((0, 4))  # position + velocity
        self.bodies = {}  # maps body identifer to index
        self.masses = np.empty((0,))
        self.springs = {}
        self.stiffness = stiffness
        self.drag = 0

    def set_time(self, t):
        self.t = t

    def add_body(self, position, velocity, identity, mass=1):
        state = np.hstack((position, velocity))
        assert state.shape[1] == self.state.shape[1]
        self.bodies[identity] = len(self.state)
        self.state = np.vstack((self.state, state))
        self.masses = np.append(self.masses, mass)
        return identity

    def remove_body(self, identity) -> None:
        i = self.bodies[identity]
        # 1. remove springs attached to body
        to_remove = [sid for sid, spring in self.springs.items() if i == spring[0] or i == spring[1]]
        for sid in to_remove:
            self.remove_spring(sid)

        # 2. Update all springs refering the last element to now refer i instead
        last = len(self.state) - 1
        self.springs = {sid: replace_index(spring, last, i) for sid, spring in self.springs.items()}

        # 3. swap last and i
        for last_identity in (identity for identity, index in self.bodies.items() if index == last):
            self.bodies[last_identity] = i

        self.state[i], self.state[-1] = self.state[-1], self.state[i]
        self.masses[i], self.masses[-1] = self.masses[-1], self.masses[i]

        # 4. remove last element
        del self.bodies[identity]
        self.state = self.state[:-1]
        self.masses = self.masses[:-1]

    def add_spring(self, identifier, a, b, length, stiffness=1) -> None:
        """Adds a spring"""
        i = self.bodies[a]
        j = self.bodies[b]
        self.springs[identifier] = (i, j, length, stiffness, self.t)

    def remove_spring(self, identifier):
        del self.springs[identifier]
    
    def iter_springs(self):
        """Iterate all spring ids and age"""
        t = self.t
        for spring_id, spring in self.springs.items():
            yield spring_id, t - spring[4]

    def delta_state(self, state: np.array, t: float):
        # compute spring forces

        positions, velocities = np.hsplit(state, 2)

        # find indices and lengths
        ii = [spring[0] for spring in self.springs.values()]
        jj = [spring[1] for spring in self.springs.values()]
        ll = [spring[2] for spring in self.springs.values()]
        ss = [spring[3] for spring in self.springs.values()]
        tt = [spring[4] for spring in self.springs.values()]
    
        # compute deltas
        di = positions[jj] - positions[ii]
        dj = positions[ii] - positions[jj]

        # compute lengths
        norms = np.linalg.norm(di, axis=-1)

        # compute spring force magnitudes
        age = self.t - np.array(tt)
        force_magnitudes = (norms - ll) * self.stiffness(np.array(ss), age)

        forces = np.zeros(positions.shape)
        forces[ii] += (di / norms[:, None]) * force_magnitudes[:, None]
        forces[jj] += (dj / norms[:, None]) * force_magnitudes[:, None]

        # add drag forces
        forces += -self.drag * velocities

        # compute acceleration
        return np.hstack((velocities, forces / self.masses[:, None]))

    def positions(self):
        positions, _ = np.hsplit(self.state, 2)
        return positions

    def step(self, dt: float):
        p = self.state
        v = self.delta_state
        t = self.t

        k1 = dt * v(p, t)
        k2 = dt * v(p + 0.5 * k1, t + 0.5 * dt)
        k3 = dt * v(p + 0.5 * k2, t + 0.5 * dt)
        k4 = dt * v(p + k3, t + dt)
        self.state = p + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        self.t += dt
