const std = @import("std");
const rl = @import("raylib");

const print = std.debug.print;
const Vec3 = rl.Vector3;
const Allocator = std.mem.Allocator;

const fade_vert_src =
\\#version 330
\\in vec3 vertexPosition;
\\in vec4 vertexColor;
\\out vec3 fragPosition;
\\out vec4 fragColor;
\\uniform mat4 matModel;
\\uniform mat4 mvp;
\\void main() {
\\  fragColor = vertexColor;
\\  fragPosition = vec3(matModel*vec4(vertexPosition, 1.0));
\\  gl_Position = mvp*vec4(vertexPosition, 1.0);
\\}
;

const fade_frag_src =
\\#version 330
\\in vec3 fragPosition;
\\in vec4 fragColor;
\\out vec4 finalColor;
\\
\\uniform float fadeDistance;
\\
\\void main() {
\\    float dist = length(fragPosition);
\\    float alpha = 1.0 - smoothstep(0.0, fadeDistance, dist);
\\    finalColor = vec4(fragColor.rgb, fragColor.a * alpha);
\\}
;

fn drawAxis(size: f32) void {
    rl.drawLine3D(.init(-size, 0, 0), .init(size, 0, 0), .red);
    rl.drawLine3D(.init(0, -size, 0), .init(0, size, 0), .green);
    rl.drawLine3D(.init(0, 0, -size), .init(0, 0, size), .blue);
}

/// perform kmeans clustering and return the class labels for the classes
const epsilon: f32 = 1e-4;
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
        const idx = rand.uintAtMost(usize, means.len - 1);
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

    // Axis fade shader
    const fade_shader = try rl.loadShaderFromMemory(fade_vert_src, fade_frag_src);
    defer fade_shader.unload();

    const axis_length: f32 = 1000.0;
    const fade_distance_loc = rl.getShaderLocation(fade_shader, "fadeDistance");
    rl.setShaderValue(fade_shader, fade_distance_loc, &axis_length, .float);

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
    const start_pos = Vec3.init(10, 10, 10);
    var camera = rl.Camera{
        .position = start_pos,
        .target = .init(0, 0, 0),
        .up = .init(0, 1, 0),
        .fovy = 45,
        .projection = .perspective,
    };

    rl.setTargetFPS(144);
    var up = camera.up.normalize();
    var forward = camera.target.subtract(camera.position).normalize();
    var right = forward.crossProduct(up).normalize();

    var converged = false;
    var iteration: u32 = 0;
    while (!rl.windowShouldClose()) {
        const dt = rl.getFrameTime();
        _ = dt;
        const scroll = rl.getMouseWheelMove();
        if (scroll != 0) {
            const move = scroll * 2;
            const dir = camera.position.subtract(camera.target).normalize();
            const pos = camera.position.subtract(dir.scale(move));
            if (pos.distance(camera.target) > 2) {
                camera.position = pos;
            }
        }

        if (rl.isMouseButtonDown(.left)) {
            var delta = rl.getMouseDelta();
            delta = delta.scale(0.2);
            const yaw = -std.math.degreesToRadians(delta.x);
            const pitch = -std.math.degreesToRadians(delta.y);

            var rotation = rl.Matrix.rotate(right, pitch);
            rotation = rotation.multiply(.rotate(up, yaw));
            // rotate the camera position and orientation vectors
            camera.position = camera.position.transform(rotation);
            up = up.transform(rotation);
            forward = forward.transform(rotation);
            right = right.transform(rotation);
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
            camera.position = start_pos;
            up = camera.up.normalize();
            forward = camera.target.subtract(camera.position).normalize();
            right = forward.crossProduct(up).normalize();
        }

        // Rendering
        rl.beginDrawing();
        rl.clearBackground(.black);

        fade_shader.activate();
        camera.begin();
        drawAxis(axis_length);
        camera.end();
        fade_shader.deactivate();


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
