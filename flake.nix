{
  description = "dev shell";
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-25.11";
  };
  outputs =
    { self, nixpkgs, ... }@inputs:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
    in
    {
      devShells.${system} = {
        default = pkgs.mkShell {
          packages = [
            pkgs.libtool
            pkgs.autoconf
            pkgs.automake

            pkgs.R
            pkgs.pandoc

            pkgs.gcc
            pkgs.gfortran
            pkgs.gnumake
            pkgs.pkg-config
            pkgs.cmake

            pkgs.hdf5
            pkgs.libxml2
            pkgs.curl
            pkgs.openssl
            pkgs.zlib
            pkgs.bzip2
            pkgs.xz
            pkgs.icu
            pkgs.fontconfig
            pkgs.freetype
          ];
          shellHook = ''
            export ACLOCAL_PATH="${pkgs.libtool}/share/aclocal''${ACLOCAL_PATH:+:$ACLOCAL_PATH}"
          '';
        };
      };
    };
}
