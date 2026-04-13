{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
  };

  outputs = { nixpkgs, ... }:
    let
      system = "x86_64-linux";
      pkgs = nixpkgs.legacyPackages.${system};

      # Minimal package set used by every CI step.
      ciPackages = with pkgs; [
        python312       # Python 3.12 runtime
        uv              # fast Python package manager
        ruff            # linter + formatter
        go-task         # task runner
        mdbook          # documentation builder
      ];

      # Local-dev-only tools layered on top of the CI packages.
      devOnlyPackages = with pkgs; [
      ];
    in
    {
      devShells.${system} = let
        shellHook = ''
          export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        '';
      in {
        # Minimal shell for CI. Invoke via `nix develop .#ci`.
        ci = pkgs.mkShell {
          packages = ciPackages;
          inherit shellHook;
        };

        # Full local-development shell. Activated by direnv via `.envrc`.
        default = pkgs.mkShell {
          packages = ciPackages ++ devOnlyPackages;
          inherit shellHook;
        };
      };
    };
}
