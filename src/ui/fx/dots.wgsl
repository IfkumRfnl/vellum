// Progress as a field of dots, in the manner of Hex's dot scope.
//
// The plate is Rune's own rounded element; this draws the dots inside it. Unlit dots are a faint
// grid that fades toward the top and bottom rows. Lit dots fill from the left, white, a little
// larger. The leading columns are live: orange, flickering as the work arrives. A slow beam
// passes over the lit dots while the job runs. When it's done every dot turns green, then
// settles back to white.
//
// The strip variant is the reader's: one row of dots, lit by how far you've read, green at the
// end.
//
// Colours are sRGB-encoded and premultiplied.

struct Dots {
    // The slot's rect in the pass's pixels.
    rect: vec4f,
    // The pass's size in pixels, device pixels per point, and 1 when the frame blends in sRGB.
    pass_size: vec4f,
    // Seconds, progress 0 to 1, seconds since finishing (< 0: running).
    state: vec4f,
    // 1 for the strip, dot pitch in points, rows, opacity.
    style: vec4f,
}

var<immediate> u: Dots;

struct Varyings {
    @builtin(position) position: vec4f,
    // Points from the slot's top-left corner.
    @location(0) point: vec2f,
}

@vertex
fn dots_vertex(@builtin(vertex_index) i: u32) -> Varyings {
    var corners = array<vec2f, 6>(
        vec2f(0.0, 0.0), vec2f(1.0, 0.0), vec2f(0.0, 1.0),
        vec2f(0.0, 1.0), vec2f(1.0, 0.0), vec2f(1.0, 1.0));
    let q = corners[i];
    let pixel = u.rect.xy + q * u.rect.zw;
    var out: Varyings;
    out.position = vec4f(
        pixel.x / u.pass_size.x * 2.0 - 1.0,
        1.0 - pixel.y / u.pass_size.y * 2.0,
        0.0, 1.0);
    out.point = q * u.rect.zw / u.pass_size.z;
    return out;
}

const WHITE: vec3f = vec3f(0.94, 0.94, 0.95);
const ORANGE: vec3f = vec3f(1.0, 0.624, 0.039);
const GREEN: vec3f = vec3f(0.188, 0.820, 0.345);

fn hash(p: vec2f) -> f32 {
    return fract(sin(dot(p, vec2f(127.1, 311.7))) * 43758.5453);
}

fn sstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

fn over(acc: ptr<function, vec4f>, color: vec3f, alpha: f32) {
    let k = clamp(alpha, 0.0, 1.0);
    (*acc) = vec4f(color * k + (*acc).rgb * (1.0 - k), k + (*acc).a * (1.0 - k));
}

@fragment
fn dots_fragment(v: Varyings) -> @location(0) vec4f {
    let t = u.state.x;
    let progress = clamp(u.state.y, 0.0, 1.0);
    let done_age = u.state.z;
    let done = done_age >= 0.0;
    let strip = u.style.x > 0.5;
    let pitch = u.style.y;
    let rows = max(i32(u.style.z + 0.5), 1);
    let size = u.rect.zw / u.pass_size.z;
    let aa = 0.8 / u.pass_size.z;

    // The grid, centred in the slot.
    let columns = max(i32(floor(size.x / pitch)), 1);
    let origin = (size - vec2f(f32(columns), f32(rows)) * pitch) * 0.5;
    let cell = (v.point - origin) / pitch;
    let i = i32(floor(cell.x));
    let j = i32(floor(cell.y));
    if (i < 0 || j < 0 || i >= columns || j >= rows) {
        return vec4f(0.0);
    }
    let center = origin + (vec2f(f32(i), f32(j)) + 0.5) * pitch;
    let distance = length(v.point - center);

    // How lit this column is: whole columns behind the front, a part at it.
    let front = progress * f32(columns);
    let column = f32(i);
    let lit = select(clamp(front - column, 0.0, 1.0), 1.0, done);
    // Columns just behind the front are live.
    let behind = front - column - 1.0;
    let live = select(exp(-max(behind, 0.0) * 0.55) * sstep(0.0, 0.6, lit), 0.0, done || strip);

    // Each live dot flickers on its own.
    let seed = hash(vec2f(column, f32(j)));
    let flicker = 0.72 + 0.28 * sin(t * (5.0 + seed * 7.0) + seed * 40.0);

    // A slow beam passing over what's done.
    let beam_at = fract(t / 2.6) * (f32(columns) + 16.0) - 8.0;
    let beam = select(exp(-pow((column - beam_at) / 3.0, 2.0)) * 0.35, 0.0, done || strip);

    var row_fade = 1.0;
    if (rows > 1) {
        let row = abs(f32(j) + 0.5 - f32(rows) * 0.5) / (f32(rows) * 0.5);
        row_fade = 1.0 - 0.7 * sstep(0.35, 0.95, row);
    }
    let unlit = select(0.10, 0.22, strip) * row_fade;
    let power = lit * mix(0.82, 1.0, row_fade) * mix(1.0, flicker, live);
    let alpha = mix(unlit, 0.96, power) + beam * lit;
    let radius = pitch * mix(0.22, 0.34, power);

    var color = mix(WHITE, ORANGE, live);
    if (done) {
        // Green on finishing, settling back to white; the strip stays green.
        let green = select(exp(-max(done_age - 0.25, 0.0) * 1.6), 1.0, strip);
        // A sweep of light across the field as it turns.
        let sweep = exp(-pow(column - done_age * f32(columns) * 1.8, 2.0) / 6.0);
        color = mix(WHITE, GREEN, clamp(green, 0.0, 1.0));
        color = mix(color, vec3f(1.0), sweep * 0.6 * select(1.0, 0.0, strip));
    }

    var acc = vec4f(0.0);
    over(&acc, color, clamp(alpha, 0.0, 1.0) * sstep(aa, -aa, distance - radius));
    var result = acc * u.style.w;
    if (u.pass_size.w < 0.5) {
        result = vec4f(pow(max(result.rgb, vec3f(0.0)), vec3f(2.2)), result.a);
    }
    return result;
}
