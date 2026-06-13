// Demonstration target for the mutation tester: true iff lo <= x <= hi.
extern "C" int in_range(int x, int lo, int hi) {
    return (x >= lo && x <= hi) ? 1 : 0;
}
