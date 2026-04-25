const std = @import("std");
const rl = @import("raylib");

const print = std.debug.print;
const Vec3 = rl.Vector3;
const Allocator = std.mem.Allocator;

fn drawAxis(sz: f32) void {
    const radius = 0.008;
    const rings = 40;
    const slices = 2;
    rl.drawCapsule(.init(-sz, 0, 0), .init(sz, 0, 0), radius, rings, slices, .red);
    rl.drawCapsule(.init(0, -sz, 0), .init(0, sz, 0), radius, rings, slices, .green);
    rl.drawCapsule(.init(0, 0, -sz), .init(0, 0, sz), radius, rings, slices, .blue);
}

/// Phi and theta are in degrees
fn spherical_to_cartesian(r: f32, phi: f32, theta: f32) Vec3 {
    const phi_rad = std.math.degreesToRadians(phi);
    const theta_rad = std.math.degreesToRadians(theta);
    return .{
        .x = r * @sin(theta_rad) * @sin(phi_rad),
        .y = r * @cos(theta_rad),
        .z = r * @sin(theta_rad) * @cos(phi_rad),
    };
}

/// perform kmeans clustering and return the class labels for the classes
const epsilon: f32 = 1e-5;
fn kmeansIteration(points: []Vec3, k: usize, means: []Vec3, labels: []usize) bool {
    for (0..labels.len) |i| {
        const p = points[i];
        var closest: usize  = 0;
        var closest_sqr_dist: f32 = std.math.inf(f32);
        for (means, 0..) |m, j| {
            const dist = p.distanceSqr(m);
            if (dist < closest_sqr_dist) {
                closest_sqr_dist = dist;
                closest = j;
            }
        }
        labels[i] = closest;
    }

    // maximization: update means
    var mean_delta: f32 = 0.0;
    for (0..k) |i| {
        var total = Vec3.zero();
        var count: usize = 0;
        for (0..points.len) |j| {
            if (labels[j] != i) continue;
            total = total.add(points[j]);
            count += 1;
        }
        if (count != 0) {
            const scale = 1.0 / @as(f32, @floatFromInt(count));
            const mean = total.scale(scale);
            mean_delta += mean.distance(means[i]);
            means[i] = mean;
        }
    }
    mean_delta /= @as(f32, @floatFromInt(k));
    return mean_delta < epsilon;
}

fn kmeansInitialize(points: []Vec3, means: []Vec3, labels: []usize, rand: std.Random) void {
    for (means) |*mean| {
        const idx = rand.uintAtMost(usize, points.len - 1);
        mean.* = points[idx];
    }
    @memset(labels, 0);
}

fn randomColors(colors: []rl.Color, rand: std.Random) void {
    for (colors) |*color| {
        color.* = .{
            .r = rand.intRangeAtMost(u8, 50, 255),
            .g = rand.intRangeAtMost(u8, 50, 255),
            .b = rand.intRangeAtMost(u8, 50, 255),
            .a = 255,
        };
    }
}

pub fn initGaussian(points: []Vec3, classes: u32, region_size: u32, rand: std.Random, alloc: Allocator) void {
    const class_prob_range: f32 = @min(1.0, 1.2 / @as(f32, @floatFromInt(classes)));
    const variance_range = @as(f32, @floatFromInt(region_size)) / 4.0;
    const mean_range: f32 = @floatFromInt(region_size);
    const means = alloc.alloc(Vec3, classes) catch unreachable;
    const variances = alloc.alloc(Vec3, classes) catch unreachable;
    const class_probs = alloc.alloc(f32, classes) catch unreachable;

    var total_prob: f32 = 1.0;
    for (0..classes) |i| {
        const rand_prob = total_prob * (0.1 + rand.float(f32)*class_prob_range);
        const prob = if (i != classes - 1) rand_prob  else total_prob;
        total_prob -= prob;
        class_probs[i] = prob;

        means[i] = .{
            .x = -mean_range/2 + rand.float(f32) * 2*mean_range,
            .y = -mean_range/2 + rand.float(f32) * 2*mean_range,
            .z = -mean_range/2 + rand.float(f32) * 2*mean_range,
        };
        variances[i] = .{
            .x = rand.float(f32) * variance_range,
            .y = rand.float(f32) * variance_range,
            .z = rand.float(f32) * variance_range,
        };
    }

    for (points) |*point| {
        const class = rand.weightedIndex(f32, class_probs);

        const mu = means[class];
        const sigma = variances[class];
        point.* = .{
            .x = @sqrt(sigma.x)*rand.floatNorm(f32) + mu.x,
            .y = @sqrt(sigma.y)*rand.floatNorm(f32) + mu.y,
            .z = @sqrt(sigma.z)*rand.floatNorm(f32) + mu.z,
        };
    }
}

/// 3D kmeans visualization
fn runKmeans(rand: std.Random, alloc: Allocator, arena: *std.heap.ArenaAllocator) !void {
    rl.setConfigFlags(.{ .msaa_4x_hint = true });
    rl.initWindow(800, 800, "clustering");
    defer rl.closeWindow();

    const axis_length: f32 = 1000.0;

    const point_count = 100;
    const point_radius = 0.1;
    const region_size = 4;
    const classes = 3;
    const k: u32 = 3;

    const points = try alloc.alloc(Vec3, point_count);
    const labels = try alloc.alloc(usize, point_count);
    const means = try alloc.alloc(Vec3, k);
    defer {
        alloc.free(points);
        alloc.free(labels);
        alloc.free(means);
    }

    initGaussian(points, classes, region_size, rand, arena.allocator());
    _ = arena.reset(.free_all);
    kmeansInitialize(points, means, labels, rand);

    const colors: []rl.Color = try alloc.alloc(rl.Color, classes);
    defer alloc.free(colors);
    randomColors(colors, rand);

    // Camera state
    var radius: f32 = 10;
    var theta: f32 = 45;
    var phi: f32 = 45;
    var camera = rl.Camera{
        .position = spherical_to_cartesian(radius, phi, theta),
        .target = .init(0, 0, 0),
        .up = .init(0, 1, 0),
        .fovy = 45,
        .projection = .perspective,
    };

    rl.setTargetFPS(144);
    var converged = false;
    var iteration: u32 = 0;

    while (!rl.windowShouldClose()) {
        const dt = rl.getFrameTime();
        _ = dt;
        const scroll = rl.getMouseWheelMove();
        if (scroll != 0) {
            var move = 0.2 * radius; 
            if (scroll > 0) move *= -1;
            radius = @max(0.2, radius + move);
            camera.position = spherical_to_cartesian(radius, phi, theta);
        }

        if (rl.isMouseButtonDown(.left)) {
            var delta = rl.getMouseDelta();
            delta = delta.scale(0.08);
            theta -= delta.y;
            phi -= delta.x;
            theta = std.math.clamp(theta, 1, 179);
            camera.position = spherical_to_cartesian(radius, phi, theta);
        }

        if (rl.isKeyPressed(.space) and !converged) {
            iteration += 1;
            converged = kmeansIteration(points, k, means, labels);
        } else if (rl.isKeyPressed(.enter)) {
            initGaussian(points, classes, region_size, rand, arena.allocator());
            _ = arena.reset(.free_all);
            kmeansInitialize(points, means, labels, rand);
            randomColors(colors, rand);
            iteration = 0;
            converged = false;
        } else if (rl.isKeyPressed(.r)) {
            // reset position
            phi = 45;
            theta = 45;
            camera.position = spherical_to_cartesian(radius, phi, theta);
        }

        // Rendering
        rl.beginDrawing();
        rl.clearBackground(.black);

        camera.begin();
        drawAxis(axis_length);
        camera.end();


        camera.begin();
        for (points, 0..) |point, i| {
            rl.drawSphere(point, point_radius, colors[labels[i]]);
        }
        for (means, 0..) |mean, i| {
            const sz = 2.5 * point_radius;
            rl.drawSphereWires(mean, sz, 10, 10, colors[i]);
        }
        camera.end();

        // UI
        const pad = 10;
        const font_size = 30;
        var cursor: i32 = pad;
        rl.drawText(rl.textFormat("k: %u", .{k}), pad, cursor, font_size, .white);
        cursor += font_size + pad;
        rl.drawText(rl.textFormat("iteration: %u", .{iteration}), pad, cursor, font_size, .white);
        cursor += font_size + pad;
        if (converged) {
            rl.drawText("Converged", pad, cursor, font_size, .green);
        }

        rl.endDrawing();
    }
}



pub fn main(init: std.process.Init) !void {
    const alloc = init.gpa;
    var seed: u64 = undefined;
    init.io.random(std.mem.asBytes(&seed));
    var prng = std.Random.DefaultPrng.init(seed);
    const rand = prng.random();

    try runKmeans(rand, alloc, init.arena);
}
