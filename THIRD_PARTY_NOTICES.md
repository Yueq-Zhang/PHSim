# Third-Party Software Notices

This repository redistributes third-party source code. The repository's
top-level project license does not replace the licenses listed below. Each
component remains subject to its own copyright notices and license terms.

| Component | Bundled location | Version | License | Upstream |
| --- | --- | --- | --- | --- |
| NewtonSim, derived from DRAMsim3 | `ext/NewtonSim` | local fork | MIT | https://github.com/umd-memsys/DRAMsim3 |
| BookSim 2.0 | `ext/booksim` | local fork | BSD-2-Clause | https://github.com/booksim/booksim2 |
| {fmt} | `ext/fmt` and `ext/NewtonSim/ext/fmt` | 5.2.2 | BSD-2-Clause | https://github.com/fmtlib/fmt |
| spdlog | `ext/spdlog` | 1.11.0 | MIT | https://github.com/gabime/spdlog |
| {fmt}, bundled by spdlog | `ext/spdlog/fmt/bundled` | bundled with spdlog 1.11.0 | MIT with binary-distribution exception | https://github.com/fmtlib/fmt |
| argparse | `ext/argparse.hpp` | copyright 2019-2022 snapshot | MIT | https://github.com/p-ranav/argparse |
| JSON for Modern C++ | `ext/json.hpp` | 3.11.3 | MIT and embedded notices | https://github.com/nlohmann/json |
| inih / INIReader | `ext/INIReader.h` and `ext/NewtonSim/ext/headers/INIReader.h` | bundled snapshots | BSD-3-Clause | https://github.com/benhoyt/inih |
| taywee/args | `ext/NewtonSim/ext/headers/args.hxx` | bundled snapshot | MIT | https://github.com/Taywee/args |
| Catch2 | `ext/NewtonSim/ext/headers/catch.hpp` | 2.7.0 | BSL-1.0 | https://github.com/catchorg/Catch2 |
| JSON for Modern C++ | `ext/NewtonSim/ext/headers/json.hpp` | 3.5.0 | MIT | https://github.com/nlohmann/json |

## License text locations

- NewtonSim: `ext/NewtonSim/LICENSE`
- BookSim 2.0: `ext/booksim/LICENSE`
- {fmt} 5.2.2: `ext/fmt/LICENSE.rst` and
  `ext/NewtonSim/ext/fmt/LICENSE.rst`
- spdlog: `ext/spdlog/LICENSE`
- spdlog's bundled {fmt}: `ext/spdlog/fmt/bundled/fmt.license.rst`
- argparse, taywee/args, and both JSON single-header copies: the complete MIT
  notice is embedded at the beginning of each redistributed header
- inih / INIReader: `ext/INIReader.LICENSE.txt` and
  `ext/NewtonSim/ext/headers/INIHLICENSE.txt`
- Catch2: `ext/NewtonSim/ext/headers/LICENSE_1_0.txt`

The dependency versions above were determined from the source files currently
bundled in this repository. If a vendored dependency is updated, this inventory
must be reviewed at the same time.
