#include "common_function.hpp"

#include <cstdlib>
#include <iostream>
#include <memory>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

std::unique_ptr<MemoryAccess> make_access(uint32_t id) {
    auto access = std::unique_ptr<MemoryAccess>(new MemoryAccess{});
    access->id = id;
    return access;
}

}  // namespace

int main() {
    MemoryAccessOwner owner;
    require(owner.outstanding() == 0, "new owner should be empty");

    MemoryAccess* first = owner.adopt(make_access(1));
    MemoryAccess* second = owner.adopt(make_access(2));
    require(owner.outstanding() == 2, "owner should track adopted requests");
    require(first->id == 1 && second->id == 2,
            "observer pointers should refer to adopted requests");

    const size_t released_slot = first->owner_slot;
    owner.release(first);
    require(owner.outstanding() == 1, "release should remove one request");

    MemoryAccess* replacement = owner.adopt(make_access(3));
    require(replacement->owner_slot == released_slot,
            "owner should reuse released slots");
    require(owner.outstanding() == 2,
            "slot reuse should preserve the outstanding count");

    auto clone = replacement->clone();
    require(clone->owner_slot == MemoryAccess::unowned_slot,
            "cloned requests must not inherit an owner slot");

    owner.release(second);
    owner.release(replacement);
    require(owner.outstanding() == 0,
            "all adopted requests should be released exactly once");

    std::cout << "RESULT PASS: MemoryAccess ownership and slot reuse\n";
    return 0;
}
