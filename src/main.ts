async function init() {
  const canvas = document.getElementById("gfx") as HTMLCanvasElement;
  if (!canvas) throw new Error("Canvas element not found");

  if (!navigator.gpu) throw new Error("WebGPU not supported");

  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error("No GPU adapter found");

  const device = await adapter.requestDevice();
  const maxTextureSize = device.limits.maxTextureDimension2D;
  const context = canvas.getContext("webgpu") as GPUCanvasContext;
  const format = navigator.gpu.getPreferredCanvasFormat();

  // ---------- BACKGROUND STARFIELD (Full-screen quad) ----------
  const bgVertices = new Float32Array([
    -1, -1,  0, 1,
     1, -1,  1, 1,
    -1,  1,  0, 0,
     1,  1,  1, 0,
  ]);

  const bgBuffer = device.createBuffer({
    size: bgVertices.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(bgBuffer, 0, bgVertices);

  // Create procedural starfield texture
  function createStarfieldTexture(): GPUTexture {
    const width = 2048;
    const height = 2048;
    const data = new Uint8Array(width * height * 4);
    
    // Background: deep space gradient
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const idx = (y * width + x) * 4;
        
        const gradient = y / height;
        const r = Math.floor(1 + gradient * 4);
        const g = Math.floor(2 + gradient * 8);
        const b = Math.floor(5 + gradient * 15);
        
        data[idx] = r;
        data[idx + 1] = g;
        data[idx + 2] = b;
        data[idx + 3] = 255;
      }
    }
    
    // Add stars
    const starCount = 6000;
    for (let i = 0; i < starCount; i++) {
      const x = Math.floor(Math.random() * width);
      const y = Math.floor(Math.random() * height);
      const idx = (y * width + x) * 4;
      
      const brightness = Math.random();
      const size = brightness > 0.95 ? 2 : (brightness > 0.85 ? 1 : 0);
      
      const colorType = Math.random();
      let r, g, b;
      if (colorType > 0.7) {
        r = 200 + Math.floor(Math.random() * 55);
        g = 220 + Math.floor(Math.random() * 35);
        b = 255;
      } else if (colorType > 0.4) {
        r = g = b = 230 + Math.floor(Math.random() * 25);
      } else {
        r = 255;
        g = 240 + Math.floor(Math.random() * 15);
        b = 200 + Math.floor(Math.random() * 40);
      }
      
      const intensity = 180 + Math.floor(brightness * 75);
      
      for (let dy = -size; dy <= size; dy++) {
        for (let dx = -size; dx <= size; dx++) {
          const sx = x + dx;
          const sy = y + dy;
          if (sx >= 0 && sx < width && sy >= 0 && sy < height) {
            const sidx = (sy * width + sx) * 4;
            const dist = Math.sqrt(dx * dx + dy * dy);
            const falloff = size === 0 ? 1 : Math.max(0, 1 - dist / (size + 1));
            
            data[sidx] = Math.min(255, data[sidx] + Math.floor(r * intensity * falloff / 255));
            data[sidx + 1] = Math.min(255, data[sidx + 1] + Math.floor(g * intensity * falloff / 255));
            data[sidx + 2] = Math.min(255, data[sidx + 2] + Math.floor(b * intensity * falloff / 255));
          }
        }
      }
    }
    
    // Add Milky Way
    for (let y = height * 0.35; y < height * 0.65; y++) {
      for (let x = 0; x < width; x++) {
        const idx = (Math.floor(y) * width + x) * 4;
        
        const centerDist = Math.abs(y - height * 0.5) / (height * 0.15);
        const milkyWayIntensity = Math.max(0, 1 - centerDist) * 
                                   (0.25 + 0.35 * Math.sin(x / width * Math.PI * 6) * 
                                    Math.cos(x / width * Math.PI * 2));
        
        if (milkyWayIntensity > 0) {
          data[idx] = Math.min(255, data[idx] + Math.floor(35 * milkyWayIntensity));
          data[idx + 1] = Math.min(255, data[idx + 1] + Math.floor(40 * milkyWayIntensity));
          data[idx + 2] = Math.min(255, data[idx + 2] + Math.floor(50 * milkyWayIntensity));
        }
      }
    }
    
    const texture = device.createTexture({
      size: [width, height],
      format: "rgba8unorm",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });

    device.queue.writeTexture(
      { texture },
      data,
      { bytesPerRow: width * 4 },
      [width, height]
    );
    
    console.log('Created starfield texture');
    return texture;
  }

  const defaultTexture = device.createTexture({
    size: [1, 1],
    format: "rgba8unorm",
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  });
  device.queue.writeTexture(
    { texture: defaultTexture },
    new Uint8Array([100, 150, 255, 255]),
    { bytesPerRow: 4 },
    [1, 1]
  );

  const sampler = device.createSampler({
    magFilter: "linear",
    minFilter: "linear",
    addressModeU: "repeat",
    addressModeV: "clamp-to-edge",
  });

  const starfieldTexture = createStarfieldTexture();

  const bgBindGroupLayout = device.createBindGroupLayout({
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.FRAGMENT,
        texture: {},
      },
      {
        binding: 1,
        visibility: GPUShaderStage.FRAGMENT,
        sampler: {},
      },
    ],
  });

  const bgBindGroup = device.createBindGroup({
    layout: bgBindGroupLayout,
    entries: [
      { binding: 0, resource: starfieldTexture.createView() },
      { binding: 1, resource: sampler },
    ],
  });

  const bgShaderCode = `
  @group(0) @binding(0) var bgTexture: texture_2d<f32>;
  @group(0) @binding(1) var bgSampler: sampler;

  struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
  }

  @vertex
  fn vs_main(@location(0) pos: vec2<f32>, @location(1) uv: vec2<f32>) -> VertexOutput {
    var output: VertexOutput;
    output.position = vec4<f32>(pos, 0.999999, 1.0);
    output.uv = uv;
    return output;
  }

  @fragment
  fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(bgTexture, bgSampler, input.uv);
  }
  `;

  const bgShaderModule = device.createShaderModule({ code: bgShaderCode });

  const bgPipeline = device.createRenderPipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [bgBindGroupLayout] }),
    vertex: {
      module: bgShaderModule,
      entryPoint: "vs_main",
      buffers: [{
        arrayStride: 4 * 4,
        attributes: [
          { shaderLocation: 0, offset: 0, format: "float32x2" },
          { shaderLocation: 1, offset: 8, format: "float32x2" },
        ],
      }],
    },
    fragment: {
      module: bgShaderModule,
      entryPoint: "fs_main",
      targets: [{ format }],
    },
    primitive: { topology: "triangle-strip" },
  });

  const skyboxBindGroup = device.createBindGroup({
    layout: skyboxBindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: starfieldTexture.createView() },
      { binding: 2, resource: sampler },
    ],
  });

  // Skybox shader
  const skyboxShaderCode = `
  struct Uniforms {
    mvp: mat4x4<f32>,
    sunDir: vec3<f32>,
    _pad1: f32,
    cameraPos: vec3<f32>,
    cloudOpacity: f32,
  };
  
  @group(0) @binding(0) var<uniform> uniforms: Uniforms;
  @group(0) @binding(1) var skyboxTexture: texture_2d<f32>;
  @group(0) @binding(2) var skyboxSampler: sampler;

  struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) worldPos: vec3<f32>,
  }

  @vertex
  fn vs_main(
    @location(0) position: vec3<f32>,
    @location(1) uv: vec2<f32>
  ) -> VertexOutput {
    var output: VertexOutput;
    output.worldPos = position;
    output.uv = uv;
    // Transform position but keep it at far plane
    var pos = uniforms.mvp * vec4<f32>(position * 0.999, 1.0);
    output.position = vec4<f32>(pos.xy, pos.w * 0.999999, pos.w);
    return output;
  }

  @fragment
  fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    var color = textureSample(skyboxTexture, skyboxSampler, input.uv);
    return color;
  }
  `;

  const skyboxShaderModule = device.createShaderModule({ code: skyboxShaderCode });

  const skyboxPipeline = device.createRenderPipeline({
    layout: skyboxPipelineLayout,
    vertex: {
      module: skyboxShaderModule,
      entryPoint: "vs_main",
      buffers: [
        {
          arrayStride: 3 * 4,
          attributes: [{ shaderLocation: 0, offset: 0, format: "float32x3" }],
        },
        {
          arrayStride: 2 * 4,
          attributes: [{ shaderLocation: 1, offset: 0, format: "float32x2" }],
        },
      ],
    },
    fragment: {
      module: skyboxShaderModule,
      entryPoint: "fs_main",
      targets: [{ format }],
    },
    primitive: { topology: "triangle-list", cullMode: "none" },
    depthStencil: {
      depthWriteEnabled: true,
      depthCompare: "less-equal",
      format: "depth24plus",
    },
  });

  // Create UI overlay for time display
  const createTimeDisplay = () => {
    const overlay = document.createElement('div');
    overlay.id = 'time-overlay';
    overlay.style.cssText = `
      position: absolute;
      top: 20px;
      right: 20px;
      background: rgba(0, 0, 0, 0.8);
      backdrop-filter: blur(10px);
      color: white;
      padding: 16px;
      border-radius: 8px;
      font-family: system-ui, -apple-system, sans-serif;
      font-size: 14px;
      z-index: 1000;
      min-width: 280px;
    `;
    
    const title = document.createElement('div');
    title.style.cssText = 'font-size: 18px; font-weight: 600; margin-bottom: 12px;';
    title.textContent = 'Real-Time Earth';
    overlay.appendChild(title);
    
    const timeDiv = document.createElement('div');
    timeDiv.id = 'time-display';
    timeDiv.style.cssText = 'margin-bottom: 8px;';
    overlay.appendChild(timeDiv);
    
    const localTimeDiv = document.createElement('div');
    localTimeDiv.id = 'local-time-display';
    localTimeDiv.style.cssText = 'font-size: 12px; color: #aaa; margin-bottom: 12px;';
    overlay.appendChild(localTimeDiv);
    
    const sunPosDiv = document.createElement('div');
    sunPosDiv.id = 'sun-position';
    sunPosDiv.style.cssText = 'font-size: 13px; color: #ffd700;';
    overlay.appendChild(sunPosDiv);
    
    const infoDiv = document.createElement('div');
    infoDiv.style.cssText = 'margin-top: 12px; padding-top: 12px; border-top: 1px solid #444; font-size: 12px; color: #aaa;';
    infoDiv.innerHTML = `
      <div style="margin-bottom: 8px;">
        • Drag to rotate<br>
        • Scroll to zoom<br>
        • Auto-rotates when idle
      </div>
      <div style="margin-top: 8px; padding-top: 8px; border-top: 1px solid #444;">
        <strong>Features:</strong><br>
        • Real-time sun position<br>
        • Live cloud data (NASA)<br>
        • Atmospheric scattering<br>
        • City lights at night<br>
        • Ocean reflections<br>
        • Day/night terminator
      </div>
    `;
    overlay.appendChild(infoDiv);
    
    document.body.appendChild(overlay);
  };

  const updateTimeDisplay = (date: Date, sunLat: number, sunLon: number) => {
    const timeDisplay = document.getElementById('time-display');
    const localTimeDisplay = document.getElementById('local-time-display');
    const sunPosDisplay = document.getElementById('sun-position');
    
    if (timeDisplay) {
      timeDisplay.textContent = `UTC: ${date.toUTCString()}`;
    }
    if (localTimeDisplay) {
      localTimeDisplay.textContent = `Local: ${date.toLocaleString()}`;
    }
    if (sunPosDisplay) {
      sunPosDisplay.textContent = `☀ Subsolar Point: ${sunLat.toFixed(2)}°N, ${sunLon.toFixed(2)}°E`;
    }
  };

  createTimeDisplay();

  // Calculate sun position from UTC time (astronomical calculation)
  function calculateSunPosition(date: Date): { lat: number; lon: number; x: number; y: number; z: number } {
    // Day of year
    const startOfYear = new Date(date.getFullYear(), 0, 0);
    const diff = date.getTime() - startOfYear.getTime();
    const dayOfYear = Math.floor(diff / 86400000);
    
    // Solar declination (latitude where sun is directly overhead)
    // Uses simplified formula with Earth's axial tilt of 23.45°
    const declination = -23.45 * Math.cos((2 * Math.PI / 365) * (dayOfYear + 10));
    const declinationRad = declination * (Math.PI / 180);
    
    // Calculate solar longitude (where sun is directly overhead)
    // The sun moves 360° in 24 hours = 15° per hour
    // At UTC midnight (0:00), the sun is at 180° longitude (opposite side of Earth from prime meridian)
    // At UTC noon (12:00), the sun is at 0° longitude (prime meridian)
    const hours = date.getUTCHours();
    const minutes = date.getUTCMinutes();
    const seconds = date.getUTCSeconds();
    const utcTime = hours + minutes / 60 + seconds / 3600;
    
    // Solar longitude: starts at -180° at midnight UTC, reaches 0° at noon UTC
    // Formula: longitude = 180° - (utcTime * 15°)
    const solarLongitude = 180 - (utcTime * 15);
    const longitudeRad = solarLongitude * (Math.PI / 180);
    
    // Convert to Cartesian coordinates (sun direction vector)
    // Note: We need to flip the Z coordinate to match WebGPU's coordinate system
    const x = Math.cos(declinationRad) * Math.cos(longitudeRad);
    const y = Math.sin(declinationRad);
    const z = Math.cos(declinationRad) * Math.sin(longitudeRad);
    
    return {
      lat: declination,
      lon: solarLongitude,
      x, y, z
    };
  }

  // ---------- SPHERE ----------
  function createSphere(radius: number, latBands: number, lonBands: number) {
    const positions: number[] = [];
    const uvs: number[] = [];
    const indices: number[] = [];

    for (let lat = 0; lat <= latBands; lat++) {
      const theta = (lat * Math.PI) / latBands;
      const sinTheta = Math.sin(theta);
      const cosTheta = Math.cos(theta);

      for (let lon = 0; lon <= lonBands; lon++) {
        const phi = (lon * 2 * Math.PI) / lonBands;
        const sinPhi = Math.sin(phi);
        const cosPhi = Math.cos(phi);

        const x = cosPhi * sinTheta;
        const y = cosTheta;
        const z = sinPhi * sinTheta;
        positions.push(radius * x, radius * y, radius * z);
        
        uvs.push(1 - lon / lonBands, lat / latBands);
      }
    }

    for (let lat = 0; lat < latBands; lat++) {
      for (let lon = 0; lon < lonBands; lon++) {
        const first = lat * (lonBands + 1) + lon;
        const second = first + lonBands + 1;
        indices.push(first, first + 1, second);
        indices.push(second, first + 1, second + 1);
      }
    }

    return {
      positions: new Float32Array(positions),
      uvs: new Float32Array(uvs),
      indices: new Uint32Array(indices),
    };
  }

  const { positions, uvs, indices } = createSphere(1.0, 60, 60);
  console.log(`Sphere created: ${positions.length / 3} vertices, ${indices.length} indices`);

  const positionBuffer = device.createBuffer({
    size: positions.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(positionBuffer, 0, positions);

  const uvBuffer = device.createBuffer({
    size: uvs.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(uvBuffer, 0, uvs);

  const indexBuffer = device.createBuffer({
    size: indices.byteLength,
    usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(indexBuffer, 0, indices);

  // ---------- UNIFORMS ----------
  // MVP (64) + sunDir (16) + cameraPos (16) + cloudOpacity (4) = 100 bytes, pad to 112
  const uniformBuffer = device.createBuffer({
    size: 112,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
        buffer: { type: "uniform" },
      },
      {
        binding: 1,
        visibility: GPUShaderStage.FRAGMENT,
        texture: {},
      },
      {
        binding: 2,
        visibility: GPUShaderStage.FRAGMENT,
        sampler: {},
      },
      {
        binding: 3,
        visibility: GPUShaderStage.FRAGMENT,
        texture: {},
      },
    ],
  });

  const pipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [bindGroupLayout],
  });

  // ---------- TEXTURE LOADING ----------
  async function loadTexture(url: string): Promise<GPUTexture> {
    const response = await fetch(url);
    const blob = await response.blob();
    const imageBitmap = await createImageBitmap(blob);

    const texture = device.createTexture({
      size: [imageBitmap.width, imageBitmap.height],
      format: "rgba8unorm",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
    });

    device.queue.copyExternalImageToTexture(
      { source: imageBitmap },
      { texture },
      [imageBitmap.width, imageBitmap.height]
    );

    return texture;
  }

  // Load real-time cloud data from NASA GIBS
  async function loadCloudTexture(): Promise<GPUTexture> {
    try {
      const now = new Date();
      const yesterday = new Date(now.getTime() - 24 * 60 * 60 * 1000); // 1 day ago for more reliable data
      const dateStr = yesterday.toISOString().split('T')[0];
      
      // NASA GIBS WMTS endpoint for MODIS Terra Corrected Reflectance
      // This provides near-real-time global imagery
      const tileMatrixSet = 'GoogleMapsCompatible_Level6';
      const layer = 'MODIS_Terra_CorrectedReflectance_TrueColor';
      
      // For simplicity, we'll use a composite tile approach
      // In production, you'd want to fetch multiple tiles and stitch them
      const baseUrl = `https://gibs.earthdata.nasa.gov/wmts/epsg4326/best/${layer}/default/${dateStr}/${tileMatrixSet}/`;
      
      console.log('Attempting to load cloud data from NASA GIBS for date:', dateStr);
      
      // Try to create a composite from multiple tiles
      // For demo, we'll use a pre-cached or fallback approach
      const fallbackUrl = 'https://gibs.earthdata.nasa.gov/wmts/epsg4326/best/MODIS_Terra_CorrectedReflectance_TrueColor/default/' + dateStr + '/250m/3/2/5.jpg';
      
      const response = await fetch(fallbackUrl);
      if (!response.ok) throw new Error('Cloud data not available');
      
      const blob = await response.blob();
      const imageBitmap = await createImageBitmap(blob);

      const texture = device.createTexture({
        size: [imageBitmap.width, imageBitmap.height],
        format: "rgba8unorm",
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
      });

      device.queue.copyExternalImageToTexture(
        { source: imageBitmap },
        { texture },
        [imageBitmap.width, imageBitmap.height]
      );

      console.log('Cloud texture loaded successfully');
      return texture;
    } catch (err) {
      console.warn('Could not load real-time cloud data:', err);
      // Return a procedural cloud texture as fallback
      return createProceduralClouds();
    }
  }

  // Create procedural clouds as fallback
  function createProceduralClouds(): GPUTexture {
    const width = 2048;
    const height = 1024;
    const data = new Uint8Array(width * height * 4);
    
    // Simple Perlin-like noise for clouds
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const idx = (y * width + x) * 4;
        
        // Multi-octave noise
        let value = 0;
        let freq = 0.01;
        let amp = 1;
        
        for (let i = 0; i < 3; i++) {
          const nx = x * freq;
          const ny = y * freq;
          value += (Math.sin(nx * 2.1) * Math.cos(ny * 1.7) + 
                   Math.sin(nx * 1.3) * Math.cos(ny * 2.9)) * amp;
          freq *= 2;
          amp *= 0.5;
        }
        
        // Normalize and apply threshold for cloud-like appearance
        value = (value + 2) / 4; // Normalize to 0-1
        const cloud = value > 0.6 ? Math.pow((value - 0.6) / 0.4, 0.5) : 0;
        
        const intensity = Math.floor(cloud * 255);
        data[idx] = intensity;     // R
        data[idx + 1] = intensity; // G
        data[idx + 2] = intensity; // B
        data[idx + 3] = Math.floor(cloud * 200); // Alpha (transparency)
      }
    }
    
    const texture = device.createTexture({
      size: [width, height],
      format: "rgba8unorm",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });

    device.queue.writeTexture(
      { texture },
      data,
      { bytesPerRow: width * 4 },
      [width, height]
    );
    
    console.log('Created procedural cloud texture');
    return texture;
  }

  let earthTexture: GPUTexture;
  try {
    earthTexture = await loadTexture("/earth_day.jpg");
    console.log("Earth texture loaded from /earth_day.jpg");
  } catch (err) {
    console.error("Failed to load earth_day.jpg, using default texture", err);
    earthTexture = defaultTexture;
  }

  // Load cloud texture
  console.log("Loading cloud data...");
  const cloudTexture = await loadCloudTexture();

  const bindGroup = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: earthTexture.createView() },
      { binding: 2, resource: sampler },
      { binding: 3, resource: cloudTexture.createView() },
    ],
  });

  // ---------- SHADERS ----------
  const shaderCode = `
  struct Uniforms {
    mvp: mat4x4<f32>,
    sunDir: vec3<f32>,
    _pad1: f32,
    cameraPos: vec3<f32>,
    cloudOpacity: f32,
  };
  
  @group(0) @binding(0) var<uniform> uniforms: Uniforms;
  @group(0) @binding(1) var earthTexture: texture_2d<f32>;
  @group(0) @binding(2) var textureSampler: sampler;
  @group(0) @binding(3) var cloudTexture: texture_2d<f32>;

  struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) worldPos: vec3<f32>,
    @location(1) uv: vec2<f32>,
  }

  @vertex
  fn vs_main(
    @location(0) position: vec3<f32>,
    @location(1) uv: vec2<f32>
  ) -> VertexOutput {
    var output: VertexOutput;
    output.worldPos = position;
    output.uv = uv;
    output.position = uniforms.mvp * vec4<f32>(position, 1.0);
    return output;
  }

  @fragment
  fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    // Sample textures
    let texColor = textureSample(earthTexture, textureSampler, input.uv);
    let cloudColor = textureSample(cloudTexture, textureSampler, input.uv);
    
    let normal = normalize(input.worldPos);
    let sunDir = normalize(uniforms.sunDir);
    let viewDir = normalize(uniforms.cameraPos - input.worldPos);
    
    // Calculate sun angle
    let sunDot = dot(normal, sunDir);
    
    // Smooth day/night terminator
    let terminatorWidth = 0.18;
    let daylight = smoothstep(-terminatorWidth, terminatorWidth, sunDot);
    
    // Subtle atmospheric scattering
    let atmosphere = max(0.0, dot(normal, viewDir));
    let atmosphereGlow = pow(atmosphere, 5.0) * 0.12;
    
    // Softer rim lighting
    let rim = 1.0 - max(0.0, dot(viewDir, normal));
    let rimGlow = pow(rim, 5.0) * 0.2 * max(0.0, sunDot + 0.15);
    
    // City lights on night side
    let nightSide = 1.0 - daylight;
    let cityLights = nightSide * 0.25 * (texColor.r * 0.3 + texColor.g * 0.4 + texColor.b * 0.3);
    // Clouds fade based on zoom (uniforms.cloudOpacity)
    let visibleCityLights = cityLights * (1.0 - cloudColor.a * 0.5 * uniforms.cloudOpacity);
    
    // Subtle ocean specular
    let halfVec = normalize(sunDir + viewDir);
    let specular = pow(max(0.0, dot(normal, halfVec)), 80.0) * 0.25 * daylight * (1.0 - cloudColor.a * 0.6 * uniforms.cloudOpacity);
    
    // Balanced ambient light
    let ambient = 0.25;
    
    // Earth surface lighting
    let earthDiffuse = ambient + (1.0 - ambient) * daylight * 0.75;
    var earthColor = texColor.rgb * earthDiffuse;
    
    // Add city lights
    earthColor += vec3<f32>(1.0, 0.85, 0.5) * visibleCityLights;
    
    // Add ocean specular
    earthColor += vec3<f32>(0.85, 0.9, 1.0) * specular;
    
    // Cloud lighting (subtle, natural look)
    let cloudLighting = smoothstep(-0.1, 0.5, sunDot);
    let cloudAmbient = 0.35;
    let cloudDiffuse = cloudAmbient + (1.0 - cloudAmbient) * cloudLighting * 0.7;
    let litCloudColor = vec3<f32>(0.92, 0.94, 0.96) * cloudDiffuse;
    
    // Dynamic cloud opacity based on zoom level
    let dynamicCloudAlpha = cloudColor.a * uniforms.cloudOpacity * 0.5;
    
    // Blend clouds over earth
    var finalColor = mix(earthColor, litCloudColor, dynamicCloudAlpha);
    
    // Subtle atmospheric effects
    finalColor += vec3<f32>(0.25, 0.4, 0.7) * atmosphereGlow;
    finalColor += vec3<f32>(0.35, 0.5, 0.8) * rimGlow;
    
    return vec4<f32>(finalColor, 1.0);
  }
  `;

  const shaderModule = device.createShaderModule({ code: shaderCode });

  // ---------- PIPELINE ----------
  const pipeline = device.createRenderPipeline({
    layout: pipelineLayout,
    vertex: {
      module: shaderModule,
      entryPoint: "vs_main",
      buffers: [
        {
          arrayStride: 3 * 4,
          attributes: [{ shaderLocation: 0, offset: 0, format: "float32x3" }],
        },
        {
          arrayStride: 2 * 4,
          attributes: [{ shaderLocation: 1, offset: 0, format: "float32x2" }],
        },
      ],
    },
    fragment: {
      module: shaderModule,
      entryPoint: "fs_main",
      targets: [{ format }],
    },
    primitive: { topology: "triangle-list", cullMode: "back" },
    depthStencil: {
      depthWriteEnabled: true,
      depthCompare: "less",
      format: "depth24plus",
    },
  });

  // ---------- RESIZE ----------
  let depthTexture: GPUTexture;
  function resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const width = Math.min(
      Math.floor(canvas.clientWidth * dpr),
      maxTextureSize,
    );
    const height = Math.min(
      Math.floor(canvas.clientHeight * dpr),
      maxTextureSize,
    );
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
      context.configure({ device, format, alphaMode: "opaque" });
      depthTexture = device.createTexture({
        size: [width, height],
        format: "depth24plus",
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
      });
    }
  }
  window.addEventListener("resize", resize);
  resize();

  // ---------- CAMERA STATE ----------
  let cameraDistance = 3.0;
  let rotationX = 0;
  let rotationY = 0;
  
  let isDragging = false;
  let lastMouseX = 0;
  let lastMouseY = 0;
  
  let lastTouchDistance = 0;
  let isPinching = false;
  
  let autoRotateSpeed = 0.001;
  let lastInteractionTime = 0;
  const autoRotateDelay = 2000;

  // ---------- MATRIX FUNCTIONS ----------
  function multiplyMat4(a: Float32Array, b: Float32Array): Float32Array {
    const out = new Float32Array(16);
    for (let i = 0; i < 4; i++) {
      for (let j = 0; j < 4; j++) {
        out[i + j * 4] = 
          a[i + 0 * 4] * b[0 + j * 4] +
          a[i + 1 * 4] * b[1 + j * 4] +
          a[i + 2 * 4] * b[2 + j * 4] +
          a[i + 3 * 4] * b[3 + j * 4];
      }
    }
    return out;
  }

  function rotationXMatrix(angle: number): Float32Array {
    const c = Math.cos(angle);
    const s = Math.sin(angle);
    return new Float32Array([
      1, 0, 0, 0,
      0, c, s, 0,
      0, -s, c, 0,
      0, 0, 0, 1,
    ]);
  }

  function rotationYMatrix(angle: number): Float32Array {
    const c = Math.cos(angle);
    const s = Math.sin(angle);
    return new Float32Array([
      c, 0, -s, 0,
      0, 1, 0, 0,
      s, 0, c, 0,
      0, 0, 0, 1,
    ]);
  }

  function translationMatrix(x: number, y: number, z: number): Float32Array {
    return new Float32Array([
      1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 1, 0,
      x, y, z, 1,
    ]);
  }

  function getMVPMatrix() {
    if (canvas.width === 0 || canvas.height === 0) {
      return new Float32Array([
        1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
      ]);
    }

    const aspect = canvas.width / canvas.height;
    const fov = Math.PI / 4;
    const near = 0.1;
    const far = 100;
    const f = 1 / Math.tan(fov / 2);

    const proj = new Float32Array([
      f / aspect, 0, 0, 0,
      0, f, 0, 0,
      0, 0, (far + near) / (near - far), -1,
      0, 0, (2 * far * near) / (near - far), 0,
    ]);

    const rotY = rotationYMatrix(rotationY);
    const rotX = rotationXMatrix(rotationX);
    const translate = translationMatrix(0, 0, -cameraDistance);
    
    const rotation = multiplyMat4(rotX, rotY);
    const view = multiplyMat4(translate, rotation);

    return multiplyMat4(proj, view);
  }

  // ---------- MOUSE CONTROLS ----------
  canvas.addEventListener("mousedown", (e) => {
    isDragging = true;
    lastInteractionTime = Date.now();
    lastMouseX = e.clientX;
    lastMouseY = e.clientY;
    canvas.style.cursor = "grabbing";
  });

  canvas.addEventListener("mousemove", (e) => {
    if (!isDragging) return;

    lastInteractionTime = Date.now();

    const deltaX = e.clientX - lastMouseX;
    const deltaY = e.clientY - lastMouseY;

    rotationY += deltaX * 0.01;
    rotationX += deltaY * 0.01;

    rotationX = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, rotationX));

    lastMouseX = e.clientX;
    lastMouseY = e.clientY;
  });

  canvas.addEventListener("mouseup", () => {
    isDragging = false;
    canvas.style.cursor = "grab";
  });

  canvas.addEventListener("mouseleave", () => {
    isDragging = false;
    canvas.style.cursor = "grab";
  });

  canvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    lastInteractionTime = Date.now();
    const zoomSpeed = 0.3;
    cameraDistance += e.deltaY * zoomSpeed * 0.01;
    cameraDistance = Math.max(1.5, Math.min(10.0, cameraDistance));
  });

  // ---------- TOUCH CONTROLS ----------
  function getTouchDistance(touches: TouchList): number {
    if (touches.length < 2) return 0;
    const dx = touches[0].clientX - touches[1].clientX;
    const dy = touches[0].clientY - touches[1].clientY;
    return Math.sqrt(dx * dx + dy * dy);
  }

  canvas.addEventListener("touchstart", (e) => {
    e.preventDefault();
    lastInteractionTime = Date.now();
    
    if (e.touches.length === 1) {
      isDragging = true;
      lastMouseX = e.touches[0].clientX;
      lastMouseY = e.touches[0].clientY;
    } else if (e.touches.length === 2) {
      isPinching = true;
      lastTouchDistance = getTouchDistance(e.touches);
    }
  });

  canvas.addEventListener("touchmove", (e) => {
    e.preventDefault();
    lastInteractionTime = Date.now();
    
    if (e.touches.length === 1 && isDragging) {
      const deltaX = e.touches[0].clientX - lastMouseX;
      const deltaY = e.touches[0].clientY - lastMouseY;

      rotationY += deltaX * 0.01;
      rotationX += deltaY * 0.01;
      rotationX = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, rotationX));

      lastMouseX = e.touches[0].clientX;
      lastMouseY = e.touches[0].clientY;
    } else if (e.touches.length === 2 && isPinching) {
      const currentDistance = getTouchDistance(e.touches);
      const delta = lastTouchDistance - currentDistance;
      cameraDistance += delta * 0.01;
      cameraDistance = Math.max(1.5, Math.min(10.0, cameraDistance));
      lastTouchDistance = currentDistance;
    }
  });

  canvas.addEventListener("touchend", (e) => {
    e.preventDefault();
    if (e.touches.length === 0) {
      isDragging = false;
      isPinching = false;
    } else if (e.touches.length === 1) {
      isPinching = false;
      isDragging = true;
      lastMouseX = e.touches[0].clientX;
      lastMouseY = e.touches[0].clientY;
    }
  });

  canvas.style.cursor = "grab";

  // ---------- RENDER LOOP ----------
  function frame() {
    if (!depthTexture) {
      requestAnimationFrame(frame);
      return;
    }

    // Auto-rotate when idle
    const timeSinceInteraction = Date.now() - lastInteractionTime;
    if (timeSinceInteraction > autoRotateDelay && !isDragging && !isPinching) {
      rotationY += autoRotateSpeed;
    }

    // Calculate real-time sun position
    const now = new Date();
    const sunPos = calculateSunPosition(now);
    
    // Update UI
    updateTimeDisplay(now, sunPos.lat, sunPos.lon);

    const mvpMatrix = getMVPMatrix();
    
    // Camera position in world space
    const camX = 0;
    const camY = 0;
    const camZ = cameraDistance;
    
    // Calculate cloud opacity based on zoom level (Google Earth style)
    // When zoomed out (distance = 3-10), clouds are visible (opacity = 1)
    // When zoomed in (distance = 1.5-2.5), clouds fade away (opacity = 0)
    const cloudOpacity = Math.min(1.0, Math.max(0.0, (cameraDistance - 1.5) / 2.0));
    
    // Write uniforms: MVP (64) + sunDir (16) + cameraPos (16) + cloudOpacity (4)
    const uniformData = new Float32Array(28);
    uniformData.set(mvpMatrix, 0);
    uniformData.set([sunPos.x, sunPos.y, sunPos.z], 16);
    uniformData.set([camX, camY, camZ], 20);
    uniformData[24] = cloudOpacity; // Cloud opacity based on zoom
    
    device.queue.writeBuffer(uniformBuffer, 0, uniformData);

    const commandEncoder = device.createCommandEncoder();
    const textureView = context.getCurrentTexture().createView();

    const renderPass = commandEncoder.beginRenderPass({
      colorAttachments: [
        {
          view: textureView,
          clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1 },
          loadOp: "clear",
          storeOp: "store",
        },
      ],
      depthStencilAttachment: {
        view: depthTexture.createView(),
        depthClearValue: 1.0,
        depthLoadOp: "clear",
        depthStoreOp: "store",
      },
    });

    // Render background first
    renderPass.setPipeline(bgPipeline);
    renderPass.setBindGroup(0, bgBindGroup);
    renderPass.setVertexBuffer(0, bgBuffer);
    renderPass.draw(4);

    // Render Earth on top
    renderPass.setPipeline(pipeline);
    renderPass.setBindGroup(0, bindGroup);
    renderPass.setVertexBuffer(0, positionBuffer);
    renderPass.setVertexBuffer(1, uvBuffer);
    renderPass.setIndexBuffer(indexBuffer, "uint32");
    renderPass.drawIndexed(indices.length);
    renderPass.end();

    device.queue.submit([commandEncoder.finish()]);
    requestAnimationFrame(frame);
  }

  requestAnimationFrame(frame);
}

init();