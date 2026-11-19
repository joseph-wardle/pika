{
  description = "Pika's build and check environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    { nixpkgs, rust-overlay, ... }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        overlays = [ rust-overlay.overlays.default ];
      };

      openusd = pkgs.python313Packages.openusd.override {
        withUsdView = true;
        withTools = true;
      };

      python = pkgs.python313.withPackages (p: [
        openusd
        p.numpy
        p.openimageio
        p.materialx
        p.pyside6
      ]);
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        packages = [
          pkgs.rust-bin.stable.latest.default
          pkgs.shader-slang
          pkgs.vulkan-tools
          pkgs.gnumake
          pkgs.ispc
          pkgs.xvfb
          python
          openusd
        ];
        buildInputs = [
          openusd
          pkgs.opensubdiv
          pkgs.onetbb
          pkgs.openimageio
          pkgs.materialx
        ];
        # ash loads libvulkan at run time, and the validation layer's manifest names its
        # library without a path.
        LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
          pkgs.vulkan-loader
          pkgs.vulkan-validation-layers
        ];
        MATERIALX = "${pkgs.materialx}";
        VK_ADD_LAYER_PATH = "${pkgs.vulkan-validation-layers}/share/vulkan/explicit_layer.d";
      };
    };
}
