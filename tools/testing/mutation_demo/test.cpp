// A *strong* test that should kill every mutant of in_range (boundaries + sides).
extern "C" int in_range(int, int, int);
int main() {
    int ok = 1;
    ok &= in_range(5, 1, 10) == 1;    // interior
    ok &= in_range(1, 1, 10) == 1;    // lower boundary (kills >= -> >)
    ok &= in_range(10, 1, 10) == 1;   // upper boundary (kills <= -> <)
    ok &= in_range(0, 1, 10) == 0;    // below       (kills && -> ||)
    ok &= in_range(11, 1, 10) == 0;   // above
    return ok ? 0 : 1;
}
