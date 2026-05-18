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
      rEnv = pkgs.rWrapper.override {
        packages = with pkgs.rPackages; [ renv ];
      };
    in
    {
      devShells.${system} = {
        default = pkgs.mkShell {
          packages = [
            pkgs.libtool
            pkgs.autoconf
            pkgs.automake
            rEnv
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
            pkgs.libuv
            pkgs.harfbuzz
            pkgs.fribidi
            pkgs.libtiff
            pkgs.libjpeg
            pkgs.libwebp
            pkgs.libpng
          ];
          shellHook = ''
            export LD_LIBRARY_PATH="${
              pkgs.lib.makeLibraryPath [
                pkgs.openssl
                pkgs.curl
                pkgs.hdf5
                pkgs.libxml2
                pkgs.zlib
                pkgs.bzip2
                pkgs.xz
                pkgs.icu
                pkgs.fontconfig
                pkgs.freetype
                pkgs.libuv
                pkgs.harfbuzz
                pkgs.fribidi
                pkgs.libtiff
                pkgs.libjpeg
                pkgs.libwebp
                pkgs.libpng
              ]
            }''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
            export ACLOCAL_PATH="${pkgs.libtool}/share/aclocal''${ACLOCAL_PATH:+:$ACLOCAL_PATH}"
          '';
        };
      };
    };
}
