The larger ODE system functions can take a while to compile, so we want to precompile the functions with PackageCompiler.

Activate Julia REPL and activate the TlFOpticalBloch module; e.g. executing the following in the module directory
```Julia
pkg> activate .
```

To create a systemimage with the ODE functions already pre-compiled activate a Julia REPL and run:
```Julia
using PackageCompiler

create_sysimage(:TlFOpticalBloch; precompile_execution_file = "compile_functions.jl", sysimage_path = "TlFOpticalBloch.so")
```
To load the systemimage start Julia with
```
julia --sysimage TlFOpticalBloch.so
```