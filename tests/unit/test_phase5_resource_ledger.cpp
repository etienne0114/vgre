#include "vgre/advanced/resource_ledger.h"
#include <iostream>
#include <vector>
#include <cassert>
#include <cstdio>

using namespace vgre::advanced;

void test_billing_basics() {
    std::cout << "Testing Resource Ledger Basics..." << std::endl;
    ResourceLedger ledger;
    ledger.reset();
    
    // Record work FOR master BY a remote node (Debit Master, Credit Node)
    ledger.recordCompute("192.168.1.50", 2.5, 8, 101, CreditDirection::DEBIT);
    
    NodeBalance bal;
    ledger.getBalance("192.168.1.50", bal);
    
    // Compute unit seconds = 2.5s * 8 cores = 20.0
    assert(bal.total_credits == 20.0);
    assert(bal.balance == 20.0);
    assert(bal.transaction_count == 1);
    
    std::cout << "  Passed (2.5s execution on 8 cores)." << std::endl;
}

void test_persistence() {
    std::cout << "Testing Resource Ledger Persistence..." << std::endl;
    
    {
        ResourceLedger writer;
        writer.recordCompute("node-a", 10.0, 1, 99, CreditDirection::DEBIT);
        writer.persist();
    }
    
    {
        ResourceLedger reader;
        reader.load();
        NodeBalance bal;
        reader.getBalance("node-a", bal);
        assert(bal.total_credits == 10.0);
        assert(bal.balance == 10.0);
    }
    
    std::cout << "  Passed (Save/Load Cycle)." << std::endl;
}

int main() {
    try {
        test_billing_basics();
        test_persistence();
        std::cout << "\nALL ResourceLedger tests PASSED." << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
        return 1;
    }
}
