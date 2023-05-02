struct CameraUniform {
    view_proj: mat4x4<f32>,
};
@group(0) @binding(0) 
var<uniform> camera: CameraUniform;
//@group(0) @binding(1) 
//var shadow_texture: texture_2d<f32>;
@group(0) @binding(2) 
var shadow_texture: texture_storage_2d<r32float, write>;
@group(0) @binding(2) 
var shadow_sampler: sampler;

struct VertexInput {
    @location(0) position: vec3<f32>,
    //@location(1) color: vec3<f32>,
};

struct VertexOutput {
    // invariant = calc pure function of vertex pos 
    @invariant @builtin(position) clip_position: vec4<f32>,
    //@location(0) color: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) signed_uv: vec2<f32>,
    @location(3) initial_sdf: f32,
};

@vertex
fn vs_main(
    model: VertexInput,
) -> VertexOutput {
    //camera.view_proj;
    var out: VertexOutput;
    //out.color = model.color;
    //out.clip_position = camera.view_proj * vec4<f32>(model.position, 1.0);

    let ray_origin = (camera.view_proj * vec4(0.0, 0.0, 0.0, 1.0)).xyz;
    let initial_sdf = sdf(ray_origin);
    out.initial_sdf = initial_sdf;


    out.clip_position = vec4<f32>(model.position, 1.0);
    out.uv = vec2<f32>(model.position.x, model.position.y) * vec2<f32>(0.5) + vec2<f32>(0.5);
    out.signed_uv = vec2<f32>(model.position.x, model.position.y);
    return out;
}

// srgb_color = ((rgb_color / 255 + 0.055) / 1.055) ^ 2.4
// TODO: try 16 bit float

// +-------+-------+
// |       |       |
// |       |       |
// +-------+-------+
// |       |       |
// |       |       |
// +-------+-------+ 

// cx = width*u;
// cy = height*v;
//
// cx/cy = (width/height) * (u/v)
// cx/cy * (v/u) = (width/height) 

// BEGIN SDF PART

fn sd_sphere(pos: vec3<f32>, r: f32) -> f32 {
    return length(pos) - r;
}

// mod space +- r around origin
fn mod_space(v: f32, r: f32) -> f32 {
    return (abs(v+r) % (2.0*r)-r);

}

fn sphere_field(p: vec3<f32>) -> f32 {
    let r = 0.5;
    return 
        sd_sphere(
            vec3<f32>(
                mod_space(p.x + r, r), 
                mod_space(p.y + r, r), 
                mod_space(p.z + r, r)), 
            0.7*r
        );

}

fn sdf(p: vec3<f32>) -> f32 {
    //return sd_juliabulb(p);
    //return sd_frame_recursive(p);
    //return sd_mandelbulb(p + vec3<f32>(0.0, 0.0, 1.0));
    //return sd_box_frame(p, vec3<f32>(1.0), 0.3333);
    //return sd_box_frame(p, vec3<f32>(1.0), 0.1);
    //return sd_menger_recursive(p);
    //return sd_menger(p);
    return sd_menger_sponge(p);
    //return sd_menger_recursive(p);
    //return sphere_field(p);
}

fn sd_juliabulb(p: vec3<f32>) -> f32 {
    var p = p;
    let power = 8.0;
    let c = vec3<f32>(0.5, 1.0, 0.7);
    // http://blog.hvidtfeldts.net/index.php/2011/09/distance-estimated-3d-fractals-v-the-mandelbulb-different-de-approximations/
	//let time = 0.0;//_Time.z*2;
    //float3 c = float3(sin(time*0.12354), sin(time*0.328432), sin(time*0.234723))*1;
    //float3 c = float3(sin(time),cos(time),sin(time*0.096234))*(sin(0.254*time)+1);
    var dr = 1.0;
    var r: f32;


    let iterations = 10;//4

    let maxRThreshold = 2.0;//1.5 //"infinity"

    //float Power = 6;//8; // Z_(n+1) = Z(n)^? + c
    var i: i32;
    for (i = 0; i < iterations; i++) {
        r = length(p);
        if (r > maxRThreshold) {
            break;
        }

		dr = pow(r, power - 1.0) * power * dr;
		
		p = pow3D_8(p, r);
        p += c;
    }
    return 0.5 * log(r) * r / dr;
}

fn pow3D_8(p: vec3<f32>, r: f32) -> vec3<f32> {
	let power = 8.0;
	// xyz -> zr,theta,phi
	var theta = acos( p.z / r );
	var phi = atan2( p.y, p.x );

	// scale and rotate
	// this is the generalized operation
	let zr = pow(r, power);
	theta = theta * power;
	phi = phi * power;

	// polar -> xyz
	//p = zr*float3(sin(theta)*cos(phi), sin(phi)*sin(theta), cos(theta));
	return zr*vec3<f32>(
        sin(theta) * cos(phi), 
        sin(phi) * sin(theta), 
        cos(theta)
    );

	//https://www.iquilezles.org/www/articles/mandelbulb/mandelbulb.htm

	//float x = p.x; float x2 = x*x; float x4 = x2*x2;
    //float y = p.y; float y2 = y*y; float y4 = y2*y2;
    //float z = p.z; float z2 = z*z; float z4 = z2*z2;

    //float k3 = x2 + z2;
    //float k2 = rsqrt( k3*k3*k3*k3*k3*k3*k3 );
    //float k1 = x4 + y4 + z4 - 6.0*y2*z2 - 6.0*x2*y2 + 2.0*z2*x2;
    //float k4 = x2 - y2 + z2;

    //p.x =  64.0*x*y*z*(x2-z2)*k4*(x4-6.0*x2*z2+z4)*k1*k2;
    //p.y = -16.0*y2*k3*k4*k4 + k1*k1;
    //p.z = -8.0*y*k4*(x4*x4 - 28.0*x4*x2*z2 + 70.0*x4*z4 - 28.0*x2*z2*z4 + z4*z4)*k1*k2;
}
fn sd_mandelbulb(p: vec3<f32>) -> f32 {

    let iterations = 4;
    let power = 8.0;
    let scale = 1.0;

    let half_power = (power - 1.0) * 0.5;
    let bailout = pow(2.0, power);

    var z = p;
    var r2 = dot(z,z);
	var dz = 1.0;

	for(var i: i32 = 0; i < iterations; i++) {
        
		dz = power * pow(r2, half_power) * dz + 1.0;
        let r = length(z);
        let theta = power * acos(z.z / r);
        let phi = power * atan2(z.y, z.x);
        z = p + pow(r, power) * 
            vec3<f32>(
                sin(theta) * cos(phi),
                sin(theta) * sin(phi), 
                cos(theta) 
            );
        
        r2 = dot(z, z);
		if (r2 > bailout) {
            break;
        }
    }
    return 0.25 * log(r2) * sqrt(r2) / dz * scale;
}


fn sd_mandelbulb2(pos: vec3<f32>) -> f32 {
    let pos = pos + vec3<f32>(0.0, 0.0, 1.0);

    let iterations = 5u;
    let bailout = 0.1*20.0;

    let power = 4.0;

	var z = pos;
	var dr = 1.0;
	var r = 0.0;
    var i: u32;
    
    i = 0u;
    loop {

		r = length(z);
		if (i >= iterations || r > bailout) {
            break;
        }
		
		// convert to polar coordinates
		var theta = acos(z.z/r);
		var phi = atan2(z.y,z.x);
        // r^(pow-1)*pow*dr + 1.0
		dr = pow(r, power - 1.0) * power * dr + 1.0;
		
		// scale and rotate the point
		let zr = pow(r, power);

		theta = theta * power;
		phi = phi * power;
		
		// convert back to cartesian coordinates
		z = pos + zr * 
            vec3<f32>(
                sin(theta) * cos(phi), 
                sin(phi) * sin(theta), 
                cos(theta)
            );

        continuing {
            i++;
        }
	}
	return 0.5*log(r)*r/dr;
}

fn sd_frame_recursive(p: vec3<f32>) -> f32 {
    let s = 0.25;

    let d = length(p);

    let x = floor(log(d) / log(s));

    let s0 = pow(s, x);
    let s1 = pow(s, x + 1.0);

    return min(
        sd_box_frame(p / s0, vec3<f32>(1.0), 0.1) * s0,
        sd_box_frame(p / s1, vec3<f32>(1.0), 0.1) * s1
    );
}
fn sd_menger_recursive(p: vec3<f32>) -> f32 {
    let s = 0.25;

    let d = length(p);

    let x = floor(log(d) / log(s));

    let s0 = pow(s, x);
    let s1 = pow(s, x + 1.0);

    return min(
        sd_menger(p / s0) * s0,
        sd_menger(p / s1) * s1
    );
    //let s = 0.25;
    //return min(
    //    min(
    //        sd_menger(p), 
    //        sd_menger(p / s) * s),
    //    min(
    //        sd_menger(p / s / s) * s * s,
    //        sd_menger(p / s / s / s) * s * s * s)
    //    );
}

    //let r = 0.5;
    //return 
    //min(
    //    min(
    //        sd_sphere(pos + vec3<f32>(0.0, 0.1, 1.0), r),
    //        sd_sphere(pos + vec3<f32>(0.0, 0.1, -1.0), r)
    //    ), 
    //    min(
    //        sd_sphere(pos + vec3<f32>(0.0, 0.0, 1.0), r),
    //        sd_sphere(pos + vec3<f32>(0.0, 0.0, -1.0), r)
    //    )
    //);

fn sdf_normal(p: vec3<f32>) -> vec3<f32> {
    let h = 0.001;
    let k = vec2<f32>(1.0, -1.0);
    return normalize( 
        k.xyy * sdf(p + k.xyy * h) + 
        k.yyx * sdf(p + k.yyx * h) + 
        k.yxy * sdf(p + k.yxy * h) + 
        k.xxx * sdf(p + k.xxx * h) 
    );

}

fn sd_cross(p: vec3<f32>) -> f32 {
    let r = 1.0;
    let px = abs(p.x);
    let py = abs(p.y);
    let pz = abs(p.z);

    let da = max(px,py);
    let db = max(py,pz);
    let dc = max(pz,px);
    return min(da, min(db, dc)) - r;
}

fn sd_menger_sponge(p: vec3<f32>) -> f32 {
    var d = sd_box(p, vec3<f32>(1.3, 0.9, 1.1));

    var s = 1.0;
    var m: u32;
    for(m = 0u; m < 4u; m++) {
        let a = ((p * s + 2.0*64.0) % 2.0) - 1.0;
        s *= 3.0;
        let r = 1.0 - 3.0 * abs(a);

        let c = sd_cross(r) / s;
        d = max(d,c);
    }

    return d;
}

fn sd_box_frame(p: vec3<f32>, b: vec3<f32>, e: f32) -> f32 {
    var p = p;
    p = abs(p)-b;
    let q = abs(p+e)-e;
    return min(min(
        length(max(vec3<f32>(p.x,q.y,q.z),vec3<f32>(0.0))) + min(max(p.x,max(q.y,q.z)),0.0),
        length(max(vec3<f32>(q.x,p.y,q.z),vec3<f32>(0.0))) + min(max(q.x,max(p.y,q.z)),0.0)),
        length(max(vec3<f32>(q.x,q.y,p.z),vec3<f32>(0.0))) + min(max(q.x,max(q.y,p.z)),0.0)
    );
}

fn sd_menger(p: vec3<f32>) -> f32 {
    let d = sd_box(p, vec3<f32>(1.0, 1.0, 1.0));
    let c = sd_cross(p * 3.0) / 3.0;
    return max(d, -c);
}

fn sd_box(p: vec3<f32>, b: vec3<f32>) -> f32 {
    let q = abs(p) - b;
    return length(max(q, vec3<f32>(0.0))) + min(max(q.x, max(q.y, q.z)),0.0);

    //q.map(|a| a.max(0.0)).magnitude() + q.x.max(q.y).max(q.z).min(0.0)
}
// END SDF PART

struct TraceOutput {
    t: f32,
    i: u32,
    last_s: f32,
    last_tol: f32,
}
fn trace_overstep(ro: vec3<f32>, rd: vec3<f32>, t_max: f32, iterations: u32, tol: f32) -> TraceOutput {
    var i = 0u;
    var t = 0.0;
    var prev_s = 0.0;
    var step_factor = 2.0;
    loop {
        let p = ro + t * rd;

        let s = sdf(p);

        if prev_s * step_factor > max(prev_s, 0.0) + max(s, 0.0) {
            t += s * (1.0 - step_factor);
            prev_s = 0.0;
            step_factor = (step_factor - 1.0) * 0.5 + 1.0;
        } 
        else {
            t += s * step_factor;
            prev_s = s;
            if (t_max < t || s < tol * t) {
                break;
            }
        }
        i++;
        if (i > iterations ) {
            break;
        }
    }
    var to: TraceOutput;
    to.t = t;
    to.i = i;
    return to;
}

// ray angle dist = t*dpdx(u)


//fn trace_overstep(ro, rd, t_max: f32, iterations: u32, tol: f32) {
//    var i = 0u;
//    var t = 0.0;
//    //loop {
//    //    let p = ro + t * rd
//
//    //}
//}

fn trace_simple(ro: vec3<f32>, rd: vec3<f32>, t_min: f32, t_max: f32, iterations: u32, tol: f32) -> TraceOutput {
    var i = 0u;
    var t = t_min;
    var s = 0.0;
    var cur_tol = 0.0;
    loop {

        let p = ro + t * rd;

        s = sdf(p);

        t += s;
        
        cur_tol = tol * t;
        if (i > iterations || t_max < t || s < cur_tol) {
            break;
        }
        i++;
    }
    var to: TraceOutput;
    to.t = t;
    to.i = i;
    to.last_s = s;
    to.last_tol = cur_tol;
    return to;
}
fn trace(ro: vec3<f32>, rd: vec3<f32>, t_min: f32, t_max: f32, iterations: u32, tol: f32) -> TraceOutput {
    return trace_simple(ro, rd, t_min, t_max, iterations, tol);
}

fn old_trace(origin: vec3<f32>, dir: vec3<f32>, t_max: f32, iterations: u32, tol: f32) -> TraceOutput {
    var t = 0.0;
    for (var i = 0u; i < iterations; i++) {
        let p = origin + t * dir;
        let s = sdf(p);
        t += s;
        let scaled_tol = tol * t;
        if (s < scaled_tol || t > t_max) { 
            var to: TraceOutput;
            to.i = i;
            to.t = t;
            return to;
        }
    }
    var to: TraceOutput;
    to.i = iterations;
    to.t = t;
    return to;
}

fn ray_color(ray_origin: vec3<f32>, ray_dir: vec3<f32>, t_min: f32, t_max: f32, ray_area: f32) -> vec4<f32> {
    let max_dist = t_max;
    var to = trace(ray_origin, ray_dir, t_min, t_max, 100u, ray_area * 0.5);//0.001);
    let i = f32(to.i);
    let t = f32(to.t);

    //let n = vec3<f32>(0.0, 0.0, 0.0);
    //let n = normalize(
    //    cross(
    //        dpdy(p),
    //        dpdx(p)
    //    )
    //);

    // try to maximize how much stuff the user sees while still hiding cutoff point
    //let fog = pow(min(t / max_dist, 1.0), 4.0);
    let fog = pow(smoothstep(0.0, max_dist, t), 10.0);
    let fog_color = vec3<f32>(0.2);
    if (fog > 0.99) {
        return vec4<f32>(fog_color, 1.0); 
    } else {

        let last_s = f32(to.last_s);
        let last_tol = f32(to.last_tol);

        let smooth_i = i + last_s/last_tol;
        let smooth_t = t + last_s * 2.0;

        //let dy = dpdy(t);
        //let dx = dpdx(t);

        let p = ray_origin + t * ray_dir;

        let n = sdf_normal(p);

        let ao = 1.0 / smooth_i;

        let sun_dir = -normalize(p);
        //let sun_dir = normalize(vec3<f32>(-0.5, 1.3, 1.0)); 
        
        let reflected = reflect(ray_dir, n);

        let sun_angle = max(0.0, dot(n, sun_dir));

        var sun_light = (0.4 * sun_angle + 0.8 * pow(max(0.0, dot(reflected, sun_dir)), 12.0)) * vec3<f32>(1.0, 0.7, 0.2);
        //if (length(sun_light) > 0.01) {
            sun_light *= shadow(
                p + last_tol * n, 
                sun_dir, 
                //last_tol * 10.0, 
                0.0,
                length(p),// - length(ray_origin) * 0.2, 
                last_tol
            );
            //sun_light *= softshadow(
            //    p + 0.001*n, 
            //    sun_dir,
            //    //last_tol * 10.0, 
            //    0.0,
            //    length(p) - length(ray_origin),
            //    32.0
            //);
        //}

        let sky_blue = vec3<f32>(135.0, 206.0, 235.0)/255.0;

        let ao_light = 0.4 * ao * sky_blue;//vec3<f32>(0.1, 0.2, 0.4);

        let light = (sun_light + ao_light)/2.0;

        //return vec4<f32>(ao, 0.0, 0.0, 1.0);

        let col = mix(light, fog_color, fog);
        return vec4<f32>(col, 1.0);
    }

}

fn softshadow(ro: vec3<f32>, rd: vec3<f32>, mint: f32, maxt: f32, k: f32) -> f32
{
    var res = 1.0;
    var t = mint;
    for(var i: i32=0; i<64; i++)
    {
        let h = sdf(ro + rd*t);
        res = min( res, k * h / t );
        if( res<0.001 || t>maxt ) {break;}
        t += clamp( h, 0.01, 0.2 );
    }
    return clamp( res, 0.0, 1.0 );
}

fn shadow(ro: vec3<f32>, rd: vec3<f32>, mint: f32, maxt: f32, tol: f32) -> f32 {
    var t = mint;
    var i: u32;
    for(i = 0u; i < 64u && t < maxt; i++ ) {
        let h = sdf(ro + rd * t);
        if(h < tol) {
            return 0.0;
        }
        t += h;
    }
    return 1.0;

    //let k = 8.0;
    //var res = 1.0;
    //var t = mint;
    //var i: u32;
    //for (i = 0u; i < 256u && t < maxt; i++) {
    //    let h = sdf(ro + rd * t);
    //    if (h < tol) {
    //        return 0.0;
    //    }
    //    res = min(res, k * h / t);
    //    t += h;
    //}
    //return res;
}


fn ray_color2(ray_origin: vec3<f32>, ray_dir: vec3<f32>) -> vec4<f32> {
    // sphere hit:
    let sphere_center = vec3<f32>(0.0, 0.0, -1.0);
    let sphere_radius = 0.5;

    let oc = ray_origin - sphere_center;
    let a = dot(ray_dir, ray_dir); // TODO: dot2, =1 iff normalized
    let half_b = dot(oc, ray_dir);
    let c = dot(oc, oc) - sphere_radius * sphere_radius;
    let disc = half_b * half_b - a * c;
    var t = 0.0;
    if (disc > 0.0) {
        t = (-half_b - sqrt(disc)) * a;
    } else {
        t = -1.0;
    }
    
    var col: vec3<f32>;
    let f = 10.0;
    if (t > 0.0) {
        let norm = normalize(ray_origin + t * ray_dir - vec3<f32>(0.0, 0.0, -1.0));
        let rdir = reflect(ray_dir, norm);
        col = 0.5*(vec3<f32>(norm)+1.0);
        let cdir = rdir;
        col *= sign(((atan2(cdir.x, cdir.z) * f + 10.0 * f) % 1.0 - 0.5) * ((acos(cdir.y) * f + 10.0 * f) % 1.0 - 0.5))*0.5+0.5;

        //let t = 0.5 * (dir.y + 1.0);
        //col += (1.0 - t) * vec3<f32>(1.0) + t * vec3<f32>(0.5, 0.7, 1.0);
        //col *= 0.5;
    } else {
        let t = 0.5 * (ray_dir.y + 1.0);
        col = (1.0 - t) * vec3<f32>(1.0) + t * vec3<f32>(0.5, 0.7, 1.0);
        let cdir = ray_dir;
        col *= sign(((atan2(cdir.x, cdir.z) * f + 10.0 * f) % 1.0 - 0.5) * ((acos(cdir.y) * f + 10.0 * f) % 1.0 - 0.5))*0.5+0.5;
    }

    var srgb_color: vec3<f32> = pow(col , vec3<f32>(2.4));
    //return vec4<f32>(
    //    in.clip_position.x/in.clip_position.y, 
    //    in.clip_position.y/in.clip_position.z, 
    //    in.clip_position.z/in.clip_position.x, 
    //    1.0
    //);
    return vec4<f32>(srgb_color, 1.0);

}

// transformation matrix:
// if component 4 = 1 => do translation + rotation
// if component 4 = 0 => do rotation
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {

    // ray: point, unit dir
    // TODO: inline aspects of this expr
    // TODO: omit norm if needed
    // TODO: camera transform.
    // img
    // cam

    let fov = 2.0; 
    let ratio = (in.clip_position.x / in.clip_position.y) / (in.uv.x / (1.0 - in.uv.y));

    let ray_dir = (camera.view_proj * vec4<f32>(normalize(
            vec3<f32>((in.uv - 0.5) * vec2<f32>(ratio, 1.0) * fov, - 1.0)
        ), 0.0)).xyz;
    let ray_origin = (camera.view_proj * vec4<f32>(vec3<f32>(0.0), 1.0)).xyz;



    let ray_area = length(fwidth(ray_dir)); // can optimize to omit sync here, also this is probably not fully correct

    

    //let t_max = 2.0;
    //let t_min = 0.0;//in.initial_sdf;
    let t_max = length(ray_origin) * 4.0;//in.initial_sdf * 5.0;
    let t_min = in.initial_sdf;
    return ray_color(ray_origin, ray_dir, t_min, t_max, ray_area);
    //return vec4(vec3(length(ray_area)), 1.0);
    //return vec4<f32>(1.0/aspect_ratio, 1.0, 0.0, 1.0);
    //return vec4<f32>(ray_dir, 1.0);
}

@compute
@workgroup_size(1) // does not need to execute in group
fn compute_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i: u32 = global_id.x;
    let j: u32 = global_id.y;

    let u = f32(i)/50.0 - 50.0;
    let v = f32(i)/50.0 - 50.0;

    let uv = vec2<f32>(u, v);
    let fov = 1.0;

    let ray_dir = normalize(vec3<f32>((uv - 0.5) * 2.0, - 1.0));
    let ray_origin = vec3<f32>(0.0, 0.0, 0.0);
    
    let to = trace(ray_origin, ray_dir, 0.0, 2.0, 100u, 0.001);
    let t: f32 = to.t;

    //textureStore(shadow_texture, vec2<u32>(i, j), vec4<f32>(t, 0.0, 0.0, 0.0));
                


}
