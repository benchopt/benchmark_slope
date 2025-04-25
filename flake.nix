{
  description = "A basic flake with a shell";
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
  inputs.systems.url = "github:nix-systems/default";
  inputs.flake-utils = {
    url = "github:numtide/flake-utils";
    inputs.systems.follows = "systems";
  };

  outputs =
    { nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = nixpkgs.legacyPackages.${system};

        # TODO: Upstream benchopt to nixpkgs
        benchopt = (
          pkgs.python3.pkgs.buildPythonPackage rec {
            pname = "benchopt";
            version = "1.6.0";
            src = pkgs.fetchPypi {
              inherit pname version;
              hash = "sha256-/89xkqQ4bmfE+rvUzuQASa1S3h3Oqe82To0McAE5UX0=";
            };

            pyproject = true;

            build-system = [
              pkgs.python3.pkgs.setuptools
              pkgs.python3.pkgs.setuptools_scm
            ];

            dependencies = with pkgs.python3.pkgs; [
              numpy
              scipy
              pandas
              matplotlib
              click
              joblib
              pygithub
              mako
              psutil
              plotly
              pyyaml
              line-profiler
              pyarrow
              pytest
            ];
          }
        );
      in
      {
        devShells.default = pkgs.mkShell {
          packages = [
            pkgs.bashInteractive
            (pkgs.python3.withPackages (ps: [
              (ps.rpy2.override {
                extraRPackages = with pkgs.rPackages; [
                  SLOPE
                ];
              })
              ps.scikit-learn
              ps.numba
              benchopt
            ]))
          ];
        };
      }
    );
}
