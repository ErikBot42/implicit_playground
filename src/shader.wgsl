
@group(0) @binding(0) 
var<uniform> camera: CameraUniform;

@group(0) @binding(1) 
var shadow_texture: texture_2d<f32>;
//@group(0) @binding(2) 
//var shadow_texture: texture_storage_2d<r32float, write>;
@group(0) @binding(2) 
var shadow_sampler: sampler;

struct VertexInput {
    @location(0) position: vec2<f32>,
    //@location(1) color: vec3<f32>,
};

struct VertexOutput {
    // invariant = calc pure function of vertex pos 
    @invariant @builtin(position) clip_position: vec4<f32>,
    //@location(0) color: vec3<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) signed_uv: vec2<f32>,
    @location(2) initial_sdf: f32,
};

@vertex
fn vs_main(
    model: VertexInput,
) -> VertexOutput {

    //camera.view_proj;
    var out: VertexOutput;
    //out.color = model.color;
    //out.clip_position = camera.view_proj * vec4<f32>(model.position, 1.0);

    let ray_origin = (camera.view_proj * vec4<f32>(0.0, 0.0, 0.0, 1.0)).xyz;
    let initial_sdf = sdf(ray_origin);
    out.initial_sdf = initial_sdf;


    out.clip_position = vec4<f32>(model.position.xy, 0.0, 1.0);
    out.uv = model.position.xy * vec2<f32>(0.5) + vec2<f32>(0.5);
    out.signed_uv = model.position.xy;
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


fn sample_shadow_texture(p: vec3<f32>) -> f32 {
    var o = normalize(abs(p));
    if (o.x > o.y && o.x > o.z) {
        o = o.zyx;
    }
    if (o.y > o.x && o.y > o.z) {
        o = o.xzy;
    }
    if (o.z > o.x && o.z > o.y) {
        o = o.yxz;
    }
    o = o/o.z;

    let sample = textureSample(shadow_texture, shadow_sampler, o.xy * 1.0 ).r; 

    return sample;
}
fn vec3_to_f32_rand(co: vec3<f32>) -> f32 {
    return fract(sin(dot(co, vec3<f32>(3.12312, 12.9898, 78.233))) * 43758.5453);
}

fn ray_color(ray_origin: vec3<f32>, ray_dir: vec3<f32>, t_min: f32, t_max: f32, ray_area: f32) -> vec4<f32> {
    let max_dist = t_max;
    var to = trace(ray_origin, ray_dir, t_min, t_max, 100u, ray_area * 0.5);//0.001);
    let i = f32(to.i);
    let t = f32(to.t);

    //let n = vec3<f32>(0.0, 0.0, 0.0);
    //let n_t = normalize(
    //    cross(
    //        dpdy(ray_origin + ray_dir * sample),
    //        dpdx(ray_origin + ray_dir * sample)
    //    )
    //);

    // try to maximize how much stuff the user sees while still hiding cutoff point
    //let fog = pow(min(t / max_dist, 1.0), 4.0);
    let fog = pow(smoothstep(0.0, max_dist, t), 3.0); //10.0
    let fog_color = vec3<f32>(0.2);
    
    let p = ray_origin + t * ray_dir;

    //var o = normalize(abs(p));
    //if (o.x > o.y && o.x > o.z) {
    //    o = o.zyx;
    //}
    //if (o.y > o.x && o.y > o.z) {
    //    o = o.xzy;
    //}
    //if (o.z > o.x && o.z > o.y) {
    //    o = o.yxz;
    //}
    //o = o/o.z;

    //let sample = textureSample(shadow_texture, shadow_sampler, o.xy * 1.0 ).r; 
    //return vec4<f32>(sample/20.0, 0.0, 0.0, 1.0);
    //return vec4<f32>(sample/20.0, 0.0, 0.0, 1.0);
    //return vec4<f32>(0.0*length(p)/5.0, sample - length(p), length(p) - sample, 1.0);
    //return vec4<f32>(sample/5.0 - 0.5, sample / 5.0, 0.0, 1.0);
    //return vec4<f32>(abs(length(p)-sample), 0.0, 0.0, 1.0);



    let t0 = ray_origin;
    let t1 = p;
    var volumetric_light = 0.0;
    let samples = 25;
    /*{
        var p = p;
        for (var i: i32 = 0; i<samples; i++) {
            let v = vec3_to_f32_rand(p);//(f32(i)+0.5)/f32(samples);
            p = mix(t0, t1, v);
            let r = length(p)/length(ray_origin);
            let sample = sample_shadow_texture(p);
            var sun_fac = 1.0/(r*r);
            if (sample < r * 0.99) { 
                sun_fac = 0.0;
            }
            volumetric_light += sun_fac/f32(samples);
        }
    }*/

    let sample = sample_shadow_texture(p);

    var sun_fac = 1.0; 
    if (sample < length(p) * 0.99) { 
        sun_fac = 0.0;
    }
    
    let sun_col = vec3<f32>(1.0, 0.7, 0.2);
    
    let fog_res_color = fog_color + sun_col*volumetric_light * 0.5;

    if (fog > 0.99) {
        return vec4<f32>(fog_res_color, 1.0); 
    } else {


        //return vec4<f32>(t, 1.0 - t, 0.0, 1.0);
        //return vec4<f32>(0.0, sample, 0.0, 1.0);




        let last_s = f32(to.last_s);
        let last_tol = f32(to.last_tol);

        let smooth_i = i + last_s/last_tol;
        let smooth_t = t + last_s * 2.0;

        //let dy = dpdy(t);
        //let dx = dpdx(t);


        //let n = sdf_normal(p);
        let n = sdf_normal_e(p, last_tol * 1.0);

        let ao = 1.0 / smooth_i;

        let sun_dir = -normalize(p);
        //let sun_dir = normalize(vec3<f32>(-0.5, 1.3, 1.0)); 
        
        let reflected = reflect(ray_dir, n);

        let sun_angle = max(0.0, dot(n, sun_dir));


        var sun_light = (0.4 * sun_angle + 0.8 * pow(max(0.0, dot(reflected, sun_dir)), 12.0)) * sun_col;
        sun_light *= sun_fac;
        sun_light += sun_col * volumetric_light;
       // if (sample < length(p) * 0.99) {
         //   sun_light *= 0.0;
        //}
        //if (length(sun_light) > 0.01) {
            //sun_light *= shadow(
            //    p + last_tol * n, 
            //    sun_dir, 
            //    //last_tol * 10.0, 
            //    0.0,
            //    length(p),// - length(ray_origin) * 0.2, 
            //    last_tol
            //);
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

        let col = mix(light, fog_res_color, fog);
        return vec4<f32>(col * sdf_color(p), 1.0);
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
    

    //let t = textureSample(shadow_texture, shadow_sampler, in.uv).r; 
    //return vec4<f32>(t/20.0, 0.0, 0.0, 0.0);
    
    
    // uv: [0..1] in xy directions

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

