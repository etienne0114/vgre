// Phase 3, Track P3-6 — Thread-block clusters + Distributed Shared Memory (DSMEM).
//
// Each CTA in a cluster runs on its own OS thread with its own shared buffer,
// registers that buffer's base, rendezvouses at cluster.sync, then reads its
// peers' shared memory through map_shared_rank (mapa.shared::cluster). The test
// proves the three faithful behaviors: (1) a CTA reading peer p's DSMEM gets
// p's data — not its own (correct cross-CTA addressing), (2) the offset is
// preserved (peer's smem[i] maps to smem[i]), and (3) cluster.sync is a real
// barrier (no CTA reads a peer before that peer has published). A 2-CTA cluster
// reduction (the stated acceptance) is checked explicitly, then generalized.

#include "vgre/core/cluster.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <thread>
#include <vector>

using vgre::cluster::Cluster;
using vgre::cluster::ClusterDim;

static int g_pass = 0, g_total = 0;
static void check(const char* name, bool ok) {
    ++g_total;
    printf(ok ? "  PASS  %s\n" : "  FAIL  %s\n", name);
    if (ok) ++g_pass;
}

// Run an N-CTA cluster reduction over DSMEM and return whether every CTA computed
// the correct sum AND every per-peer read matched the peer's published value.
// Each CTA r publishes smem = {(r+1)*10, r}; the offset-1 slot lets us verify the
// mapped address preserves the byte offset (peer's smem[1] must read back as r).
static bool runClusterReduction(unsigned N) {
    Cluster cl(N);
    std::vector<long>  partial(N, -1);
    std::vector<int>   peerOk(N, 0);
    // Stagger publication so a broken barrier (reading before publish) is caught:
    // higher ranks sleep before registering/publishing.
    std::vector<std::thread> ts;
    ts.reserve(N);
    for (unsigned r = 0; r < N; ++r) {
        ts.emplace_back([&, r] {
            int smem[2];
            if (r) std::this_thread::sleep_for(std::chrono::milliseconds(2 * r));
            cl.register_smem(r, smem);
            smem[0] = (int)((r + 1) * 10);
            smem[1] = (int)r;
            cl.sync();                                   // all CTAs published

            long sum = 0;
            bool ok = true;
            for (unsigned p = 0; p < N; ++p) {
                int v0 = vgre::cluster::dsmem_load(cl, r, &smem[0], p);   // peer p's smem[0]
                int v1 = vgre::cluster::dsmem_load(cl, r, &smem[1], p);   // peer p's smem[1]
                sum += v0;
                if (v0 != (int)((p + 1) * 10) || v1 != (int)p) ok = false; // wrong peer / bad offset
            }
            partial[r] = sum;
            peerOk[r]  = ok ? 1 : 0;
            cl.sync();                                   // hold smem alive until all reads done
        });
    }
    for (auto& t : ts) t.join();

    long expect = 0;
    for (unsigned p = 0; p < N; ++p) expect += (long)((p + 1) * 10);
    bool allSum = true, allPeer = true;
    for (unsigned r = 0; r < N; ++r) {
        if (partial[r] != expect) allSum = false;
        if (!peerOk[r])           allPeer = false;
    }
    return allSum && allPeer;
}

int main() {
    printf("=== Thread-block clusters + Distributed Shared Memory (P3-6) ===\n");

    // (0) map_shared_rank pointer arithmetic: self-map is identity; the byte offset
    //     into the box is preserved across the retarget.
    {
        Cluster cl(2);
        int a[4] = {0,0,0,0}, b[4] = {0,0,0,0};
        cl.register_smem(0, a);
        cl.register_smem(1, b);
        bool ok = cl.map_shared_rank(0, &a[0], 0) == &a[0]          // self → identity
               && cl.map_shared_rank(0, &a[2], 1) == &b[2]          // peer, offset preserved
               && cl.map_shared_rank(1, &b[3], 0) == &a[3];
        check("map_shared_rank preserves offset and retargets to the peer CTA", ok);
    }

    // (1) The stated acceptance: a 2-CTA cluster reduction reads a peer's DSMEM.
    {
        Cluster cl(2);
        int s0[1], s1[1];
        long r0 = 0, r1 = 0; int cross0 = 0, cross1 = 0;
        std::thread t0([&]{
            cl.register_smem(0, s0); s0[0] = 7;
            cl.sync();
            cross0 = vgre::cluster::dsmem_load(cl, 0, &s0[0], 1);    // read CTA1's value
            r0 = s0[0] + (long)cross0;
            cl.sync();
        });
        std::thread t1([&]{
            std::this_thread::sleep_for(std::chrono::milliseconds(3)); // publish late
            cl.register_smem(1, s1); s1[0] = 35;
            cl.sync();
            cross1 = vgre::cluster::dsmem_load(cl, 1, &s1[0], 0);    // read CTA0's value
            r1 = s1[0] + (long)cross1;
            cl.sync();
        });
        t0.join(); t1.join();
        bool ok = (cross0 == 35) && (cross1 == 7)        // each read the PEER (not itself)
               && (r0 == 42) && (r1 == 42);              // 7 + 35, despite the late publish
        check("2-CTA cluster reduction reads peer DSMEM correctly (barrier ordered)", ok);
    }

    // (2) Generalize: 4- and 8-CTA cluster reductions over DSMEM.
    check("4-CTA cluster DSMEM reduction == sum of all peers", runClusterReduction(4));
    check("8-CTA cluster DSMEM reduction == sum of all peers", runClusterReduction(8));

    // (2b) DSMEM *write*: each CTA writes a value into its right neighbor's shared
    //      memory via dsmem_store; after the barrier every CTA reads the value its
    //      left neighbor deposited. Proves cross-CTA writes (not just reads).
    {
        const unsigned N = 4;
        Cluster cl(N);
        std::vector<int> got(N, -1);
        std::vector<std::thread> ts;
        for (unsigned r = 0; r < N; ++r) {
            ts.emplace_back([&, r] {
                int inbox = 0;
                cl.register_smem(r, &inbox);
                cl.sync();                                   // all inboxes registered
                unsigned right = (r + 1) % N;
                vgre::cluster::dsmem_store(cl, r, &inbox, right, (int)(100 + r));  // write peer's inbox
                cl.sync();                                   // all writes landed
                got[r] = inbox;                              // value my left neighbor wrote
                cl.sync();
            });
        }
        for (auto& t : ts) t.join();
        bool ok = true;
        for (unsigned r = 0; r < N; ++r) {
            unsigned left = (r + N - 1) % N;
            if (got[r] != (int)(100 + left)) ok = false;
        }
        check("dsmem_store writes into a peer CTA's shared memory", ok);
    }

    // (3) Cluster rank ↔ coordinate round-trips over a 2×2×2 cluster shape.
    {
        ClusterDim d{2, 2, 2};
        bool ok = (d.size() == 8);
        for (unsigned cz = 0; cz < d.z && ok; ++cz)
            for (unsigned cy = 0; cy < d.y && ok; ++cy)
                for (unsigned cx = 0; cx < d.x && ok; ++cx) {
                    unsigned rank = vgre::cluster::cluster_rank(d, cx, cy, cz);
                    unsigned ox, oy, oz;
                    vgre::cluster::cluster_coord(d, rank, ox, oy, oz);
                    if (ox != cx || oy != cy || oz != cz) ok = false;
                }
        check("cluster rank ↔ 3-D coordinate round-trips (2x2x2)", ok);
    }

    printf("\n%d / %d passed\n", g_pass, g_total);
    return (g_pass == g_total) ? 0 : 1;
}
