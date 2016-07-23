package = "torchnet-vision"
version = "scm-1"

source = {
   url = "git://github.com/Cadene/torchnet-vision.git"
}

description = {
   summary = "Plugin vision for Torchnet",
   detailed = [[
   Various abstractions for vision processing.
   ]],
   homepage = "https://github.com/Cadene/torchnet-vision",
   license = "BSD"
}

dependencies = {
   "lua >= 5.1",
   "torch >= 7.0",
   "argcheck >= 1.0",
   "tds >= 1.0",
   "image >= 1.0"
}

build = {
   type = "cmake",
   variables = {
      CMAKE_BUILD_TYPE="Release",
      LUA_PATH="$(LUADIR)",
      LUA_CPATH="$(LIBDIR)"
   }
}
