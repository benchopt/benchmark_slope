{
  config,
  lib,
  pkgs,
  ...
}:

let
  conda = pkgs.conda.override {
    # Conda packages are ordinary Linux binaries, so they must remain inside
    # the FHS environment provided by this package on NixOS.
    # Benchopt 1.9.1 mistakes installations below the checkout for editable
    # installs when it creates test environments.
    installationPath = "~/.local/state/devenv/benchmark_slope/conda";
    extraPkgs = [
      pkgs.git
      pkgs.which
    ];
  };
  condaShell = lib.getExe conda;
in
{
  packages = [ conda ];

  env = {
    CONDA_ENVS_PATH = "${config.devenv.state}/conda/envs";
    CONDA_PKGS_DIRS = "${config.devenv.state}/conda/pkgs";
    PYTHONNOUSERSITE = "1";
  };
  unsetEnvVars = [ "PYTHONPATH" ];

  scripts.benchopt-setup = {
    description = "Install or update Benchopt in the benchmark-specific Conda base environment";
    exec = ''
      exec ${condaShell} -c '
        set -e
        conda activate base
        conda config --system --remove-key channels 2>/dev/null || true
        conda config --system --add channels conda-forge
        conda config --system --set channel_priority strict
        python -m pip install --upgrade benchopt
      '
    '';
  };

  scripts.benchopt = {
    description = "Run Benchopt in the NixOS-compatible Conda environment";
    exec = ''
      exec ${condaShell} -c 'set -e; conda activate base; export SHELL=/bin/bash; exec benchopt "$@"' benchopt "$@"
    '';
  };
}
