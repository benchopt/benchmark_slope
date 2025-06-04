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

        download = (
          pkgs.python3.pkgs.buildPythonPackage rec {
            pname = "download";
            version = "0.3.5";
            pyproject = true;

            src = pkgs.fetchFromGitHub {
              owner = "choldgraf";
              repo = "download";
              rev = "v${version}";
              hash = "sha256-socNW0PhoIj0crULeTpxULhKZGCo3QpypyScFdkX4A0=";
            };

            build-system = with pkgs.python3.pkgs; [
              setuptools
            ];

            dependencies = with pkgs.python3.pkgs; [
              tqdm
              six
              requests
            ];

            pythonImportsCheck = [
              "download"
            ];
          }
        );

        libsvmdata = (
          pkgs.python3.pkgs.buildPythonPackage rec {
            pname = "libsvmdata";
            version = "unstable-2025-04-29";
            pyproject = true;

            src = pkgs.fetchFromGitHub {
              owner = "mathurinm";
              repo = "libsvmdata";
              rev = "1533b6de47bbdef6f8df9ae78b3226d473965416";
              hash = "sha256-xWXiTyc6UNIS4zRstY0Yw4hl+qeqal+o7r0BrGudOIE=";
            };

            build-system = with pkgs.python3.pkgs; [
              setuptools
              wheel
            ];

            dependencies = with pkgs.python3.pkgs; [
              download
              numpy
              scikit-learn
              scipy
            ];

            pythonImportsCheck = [
              "libsvmdata"
            ];
          }
        );

        slopepath = (
          pkgs.python3.pkgs.buildPythonPackage rec {
            pname = "slopepath";
            version = "1.0.0";
            src = pkgs.fetchFromGitHub {
              inherit pname version;
              owner = "jolars";
              repo = "slope-path";
              rev = "20d2bab31492b835bf31c80533a927c792b43849";
              hash = "sha256-WVbT0fW/lhBoJgyozl/lQy+i/rH9mX+0E5SwdAUsRhw=";
            };

            pyproject = true;

            build-system = [
              pkgs.python3.pkgs.setuptools
            ];

            dependencies = with pkgs.python3.pkgs; [
              numba
              numpy
              scikit-learn
            ];
          }
        );

        # TODO: Upstream skglm to nixpkgs
        skglm = (
          pkgs.python3.pkgs.buildPythonPackage rec {
            pname = "skglm";
            version = "0.4";
            src = pkgs.fetchPypi {
              inherit pname version;
              hash = "sha256-EtItwK7z92u0Ps4WvrfY/zKyr9UsunIi098+OtWWYds=";
            };

            pyproject = true;

            build-system = [
              pkgs.python3.pkgs.setuptools
            ];

            dependencies = with pkgs.python3.pkgs; [
              numba
              numpy
              scikit-learn
              scipy
            ];
          }
        );

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

        # TODO: Upstream sortedl1 to nixpkgs
        sortedl1 = (
          pkgs.python3.pkgs.buildPythonPackage rec {
            pname = "sortedl1";
            version = "1.1.0";
            pyproject = true;

            src = pkgs.fetchPypi {
              inherit pname version;
              hash = "sha256-bon1d6r18eayuqhhK8zAckFWGSilX3eUc213HSeO2dQ=";
            };

            dontUseCmakeConfigure = true;

            build-system = [
              pkgs.python3.pkgs.scikit-build-core
              pkgs.python3.pkgs.pybind11
              pkgs.cmake
              pkgs.ninja
            ];

            dependencies = with pkgs.python3.pkgs; [
              numpy
              scikit-learn
              scipy
              furo
              sphinx-copybutton
              myst-parser
              pytest
            ];

            disabledTests = [
              "test_cdist"
            ];

            pythonImportsCheck = [
              "sortedl1"
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
              ps.appdirs
              libsvmdata
              benchopt
              sortedl1
              slopepath
              skglm
            ]))
          ];
        };
      }
    );
}
