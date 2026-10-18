#include <fuse/version.h>
#include <cstring>
#include <iostream>
int main() {
    if (std::strcmp(fuse::version(), FUSE_EXPECTED_VERSION) != 0)
        return 1;
    std::cout << "Native library version " << fuse::version() << '\n';
}
