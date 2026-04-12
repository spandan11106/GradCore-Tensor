#pragma once

#define GRADIENTCORE_VERSION_MAJOR 1
#define GRADIENTCORE_VERSION_MINOR 0
#define GRADIENTCORE_VERSION_PATCH 0

#define GRADIENTCORE_VERSION \
    (GRADIENTCORE_VERSION_MAJOR * 10000 + \
     GRADIENTCORE_VERSION_MINOR * 100 + \
     GRADIENTCORE_VERSION_PATCH)

#define GRADIENTCORE_VERSION_STRING "1.0.0"

namespace gradient {

    inline const char* version() { return GRADIENTCORE_VERSION_STRING; }

    inline int version_major() { return GRADIENTCORE_VERSION_MAJOR; }

    inline int version_minor() { return GRADIENTCORE_VERSION_MINOR; }

    inline int version_patch() { return GRADIENTCORE_VERSION_PATCH; }
}
