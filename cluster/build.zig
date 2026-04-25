const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const raylib_dep = b.dependency("raylib_zig", .{
        .target = target,
        .optimize = optimize,
        .linkage = .dynamic,
    });
    const raylib = raylib_dep.module("raylib");
    const libraylib = raylib_dep.artifact("raylib");

    const exe = b.addExecutable(.{
        .name = "cluster",
        .use_llvm = true,
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            // same as exe.addImport()
            .imports = &.{
                .{ .name = "raylib", .module = raylib },
            },
        }),
    });
    exe.root_module.linkLibrary(libraylib);

    b.installArtifact(exe);

    const run_step = b.step("run", "Run the app");

    const run_cmd = b.addRunArtifact(exe);
    run_step.dependOn(&run_cmd.step);
    run_cmd.step.dependOn(b.getInstallStep());

    if (b.args) |args| {
        run_cmd.addArgs(args);
    }
}
