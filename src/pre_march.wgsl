
@group(0) @binding(0) 
var pre_march_texture: texture_storage_2d<rg32float, write>;

@group(0) @binding(1) 
var<uniform> camera: CameraUniform;

@compute
@workgroup_size(1) // does not need to execute in group
fn compute_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    //let uv = vec2<f32>(global_id.xy)/vec2<f32>(textureDimensions(shadow_texture).xy);

    //let ray_dir = normalize(vec3<f32>(uv, - 1.0));
    //let ray_origin = vec3<f32>(0.0, 0.0, 0.0);
    //
    //let to = trace(
    //    ray_origin, 
    //    ray_dir, 
    //    0.1 * length((camera.view_proj * vec4<f32>(vec3<f32>(0.0), 1.0)).xyz), 
    //    20.0, 
    //    200u, 
    //    0.001
    //);
    //let t: f32 = to.t;

    let dim = textureDimensions(pre_march_texture);
    
    let uv = (vec2<f32>(global_id.xy) + vec2<f32>(0.5))/vec2<f32>(dim.xy);

    let ray_angle_radians = 1.0/f32(max(dim.x, dim.y));

    let fov = 2.0; 
    let ratio = 16.0/9.0;//(in.clip_position.x / in.clip_position.y) / (in.uv.x / (1.0 - in.uv.y));
    let ray_dir = (camera.view_proj * vec4<f32>(normalize(
            vec3<f32>((uv - 0.5) * vec2<f32>(ratio, 1.0) * fov, - 1.0)
        ), 0.0)).xyz;
    let ray_origin = (camera.view_proj * vec4<f32>(vec3<f32>(0.0), 1.0)).xyz;
    let ray_area = 2.0*ray_angle_radians;//0.05;//length(fwidth(ray_dir)); // can optimize to omit sync here, also this is probably not fully correct

    //let t_max = length(ray_origin) * 4.0;//in.initial_sdf * 5.0;
    //let t_min = 0.0;//in.initial_sdf;


    let to = trace(
        ray_origin, 
        ray_dir, 
        0.0,
        20.0, 
        200u, 
        ray_area
    );
    let t = max(to.t - abs(to.last_tol) * 1.0, 0.0);
    let i = f32(to.i) + to.last_s/to.last_tol;

    textureStore(pre_march_texture, global_id.xy, vec4<f32>(t, i, 1.0, 1.0));
}
