#include <fuse/version.h>
#include <fuse/build_config.h>
namespace fuse {
const char* version() noexcept {
    return FUSE_VERSION;
}
}  // namespace fuse
