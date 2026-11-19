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
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        packages = [
          pkgs.rust-bin.stable.latest.default
          pkgs.shader-slang
          pkgs.vulkan-tools
        ];
        # ash loads libvulkan at run time, and the validation layer's manifest names its
        # library without a path.
        LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
          pkgs.vulkan-loader
          pkgs.vulkan-validation-layers
        ];
        VK_ADD_LAYER_PATH = "${pkgs.vulkan-validation-layers}/share/vulkan/explicit_layer.d";
      };
    };
}
