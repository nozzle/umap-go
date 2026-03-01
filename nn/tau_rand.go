package nn

// Tausworthe PRNG — matches pynndescent/utils.py tau_rand_int and tau_rand.
//
// The Python implementation uses a 3-element int64 state (Tausworthe combined
// generator) with XOR-shift operations. Each call to TauRandInt returns a
// positive int64, and TauRand returns a float64 in [0, 1).
//
// This is NOT used for production randomness. It is used exclusively inside
// NN-Descent to match Python's behavior exactly when replaying recorded
// random sequences.

const (
	tauwortheMax = 0x7FFFFFFF // 2^31 - 1
)

// TauRandState is the 3-element state for the Tausworthe PRNG.
type TauRandState [3]int64

// TauRandInt generates a random int64 from the Tausworthe PRNG state.
// Matches pynndescent tau_rand_int().
func TauRandInt(state *TauRandState) int64 {
	state[0] = (((state[0] & 4294967294) << 12) & 0xFFFFFFFF) ^ ((((state[0] << 13) & 0xFFFFFFFF) ^ state[0]) >> 19)
	state[1] = (((state[1] & 4294967288) << 4) & 0xFFFFFFFF) ^ ((((state[1] << 2) & 0xFFFFFFFF) ^ state[1]) >> 25)
	state[2] = (((state[2] & 4294967280) << 17) & 0xFFFFFFFF) ^ ((((state[2] << 3) & 0xFFFFFFFF) ^ state[2]) >> 11)
	return state[0] ^ state[1] ^ state[2]
}

// TauRand generates a random float64 in [0, 1) from the Tausworthe PRNG state.
// Matches pynndescent tau_rand().
func TauRand(state *TauRandState) float64 {
	v := TauRandInt(state)
	return float64(v&int64(tauwortheMax)) / float64(tauwortheMax)
}

// TauRandIntRange generates a random int in [0, n) using the Tausworthe PRNG.
// Python uses np.abs(tau_rand_int(rng_state)) % n, so we use the absolute value.
func TauRandIntRange(state *TauRandState, n int) int {
	v := TauRandInt(state)
	if v < 0 {
		v = -v
	}
	return int(v) % n
}
