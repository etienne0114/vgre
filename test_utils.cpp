#include <iostream>
#include <filesystem>
#include "vgre/common/system_utils.h"

int main() {
    std::cout << "Home: " << vgre::common::getHomeDirectory() << std::endl;
    std::cout << "Cache Root: " << vgre::common::getCacheRoot() << std::endl;
    std::cout << "Binary Path: " << vgre::common::getBinaryPath().string() << std::endl;
    std::cout << "Include Dir: " << vgre::common::findIncludeDir() << std::endl;
    std::cout << "Compiler Path: " << vgre::common::findCompilerPath() << std::endl;
    return 0;
}
