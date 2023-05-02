// Sdf functions and things that purely depend on sdf


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
    return sd_frame_recursive(p);
    //return sd_mandelbulb(p + vec3<f32>(0.0, 0.0, 1.0));
    //return sd_box_frame(p, vec3<f32>(1.0), 0.3333);
    //return sd_box_frame(p, vec3<f32>(1.0), 0.1);
    //return sd_menger_recursive(p);
    //return sd_menger(p);
    //return sd_menger_sponge(p);
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
