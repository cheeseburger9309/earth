import './style.css'

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

  context.configure({
    device,
    format,
    alphaMode: "opaque",
  });

  // =================================================================
  // 1. SHARED RESOURCES
  // =================================================================
  
  const defaultTexture = device.createTexture({
    size: [1, 1],
    format: "rgba8unorm",
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  });
  device.queue.writeTexture(
    { texture: defaultTexture },
    new Uint8Array([0, 0, 10, 255]), // Deep black default
    { bytesPerRow: 4 },
    [1, 1]
  );

  const sampler = device.createSampler({
    magFilter: "linear",
    minFilter: "linear",
    addressModeU: "repeat",
    addressModeV: "clamp-to-edge",
  });

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

  // =================================================================
  // 2. NEW: SKYBOX (Milky Way)
  // =================================================================
  
  // 2a. Geometry
  function createSkyboxSphere(radius: number, segments: number) {
    const positions: number[] = [];
    const uvs: number[] = [];
    const indices: number[] = [];

    for (let lat = 0; lat <= segments; lat++) {
      const theta = (lat * Math.PI) / segments;
      const sinTheta = Math.sin(theta);
      const cosTheta = Math.cos(theta);

      for (let lon = 0; lon <= segments; lon++) {
        const phi = (lon * 2 * Math.PI) / segments;
        const sinPhi = Math.sin(phi);
        const cosPhi = Math.cos(phi);

        const x = cosPhi * sinTheta;
        const y = cosTheta;
        const z = sinPhi * sinTheta;
        positions.push(radius * x, radius * y, radius * z);
        
        // 1.0 - lon to flip texture inside out
        uvs.push(1.0 - lon / segments, lat / segments);
      }
    }

    for (let lat = 0; lat < segments; lat++) {
      for (let lon = 0; lon < segments; lon++) {
        const first = lat * (segments + 1) + lon;
        const second = first + segments + 1;
        indices.push(first, first + 1, second);
        indices.push(second, first + 1, second + 1);
      }
    }

    return {
      positions: new Float32Array(positions),
      uvs: new Float32Array(uvs),
      indices: new Uint32Array(indices),
      count: indices.length 
    };
  }

  // Define buffers in main scope (Fixes ReferenceError)
  const skyboxGeo = createSkyboxSphere(100.0, 64);
  const skyboxPosBuffer = device.createBuffer({ size: skyboxGeo.positions.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(skyboxPosBuffer, 0, skyboxGeo.positions);
  const skyboxUvBuffer = device.createBuffer({ size: skyboxGeo.uvs.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(skyboxUvBuffer, 0, skyboxGeo.uvs);
  const skyboxIdxBuffer = device.createBuffer({ size: skyboxGeo.indices.byteLength, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(skyboxIdxBuffer, 0, skyboxGeo.indices);

  // 2b. Pipeline
  const skyboxUniformBuffer = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  let milkyWayTexture = defaultTexture; 

  const skyboxBindGroupLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: "uniform" } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: {} },
      { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
    ],
  });

  // Mutable bind group
  let skyboxBindGroup = device.createBindGroup({
    layout: skyboxBindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: skyboxUniformBuffer } },
      { binding: 1, resource: milkyWayTexture.createView() },
      { binding: 2, resource: sampler },
    ],
  });

  // CINEMATIC SKYBOX SHADER (Fixes Pixelation)
  const skyboxShaderCode = `
  struct Uniforms { mvp: mat4x4<f32> };
  @group(0) @binding(0) var<uniform> u: Uniforms;
  @group(0) @binding(1) var t: texture_2d<f32>;
  @group(0) @binding(2) var s: sampler;

  struct Out { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> }

  @vertex
  fn vs(@location(0) pos: vec3<f32>, @location(1) uv: vec2<f32>) -> Out {
    var mvp = u.mvp;
    mvp[3] = vec4<f32>(0.0, 0.0, 0.0, 1.0); // Remove translation (Skybox mode)
    let p = mvp * vec4<f32>(pos, 1.0);
    // Force Z to max depth
    return Out(vec4<f32>(p.xy, p.w * 0.99999, p.w), uv);
  }

  @fragment
  fn fs(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    let raw = textureSample(t, s, uv).rgb;
    
    // 1. Gamma Correction (Darkens greys, brightens lights)
    let contrast = pow(raw, vec3<f32>(2.2));
    
    // 2. Black Crush (Hides artifacts)
    let clean = max(vec3<f32>(0.0), contrast - vec3<f32>(0.02));
    
    // 3. Cinematic Blue Tint
    let graded = clean * vec3<f32>(1.0, 1.05, 1.2);

    return vec4<f32>(graded, 1.0);
  }
  `;

  const skyboxPipeline = device.createRenderPipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [skyboxBindGroupLayout] }),
    vertex: {
      module: device.createShaderModule({ code: skyboxShaderCode }),
      entryPoint: 'vs',
      buffers: [
        { arrayStride: 12, attributes: [{shaderLocation:0, offset:0, format:"float32x3"}] }, 
        { arrayStride: 8, attributes: [{shaderLocation:1, offset:0, format:"float32x2"}] }
      ]
    },
    fragment: {
      module: device.createShaderModule({ code: skyboxShaderCode }),
      entryPoint: 'fs',
      targets: [{ format }]
    },
    primitive: { cullMode: 'front' }, // Inside sphere
    depthStencil: { depthWriteEnabled: false, depthCompare: 'less-equal', format: 'depth24plus' }
  });

  // Async Load
  async function loadMilkyWay() {
    try {
      console.log("Loading Milky Way...");
      const tex = await loadTexture("https://cdn.eso.org/images/publicationjpg/eso0932a.jpg");
      milkyWayTexture = tex;
      
      skyboxBindGroup = device.createBindGroup({
        layout: skyboxBindGroupLayout,
        entries: [
          { binding: 0, resource: { buffer: skyboxUniformBuffer } },
          { binding: 1, resource: milkyWayTexture.createView() },
          { binding: 2, resource: sampler },
        ],
      });
      console.log("Milky Way loaded!");
    } catch(e) { console.warn("Milky Way failed", e); }
  }
  loadMilkyWay();

  // =================================================================
  // 3. ORIGINAL STARS, EARTH, AND LOGIC (Preserved)
  // =================================================================

  function createStarfieldSphere() {
    const stars: number[] = [];
    const colors: number[] = [];
    const starCount = 8000;
    
    for (let i = 0; i < starCount; i++) {
      const theta = Math.acos(2 * Math.random() - 1);
      const phi = Math.random() * Math.PI * 2;
      const x = Math.sin(theta) * Math.cos(phi);
      const y = Math.sin(theta) * Math.sin(phi);
      const z = Math.cos(theta);
      stars.push(x, y, z);
      
      const colorType = Math.random();
      const brightness = 0.6 + Math.random() * 0.4;
      if (colorType > 0.7) {
        colors.push(0.8 * brightness, 0.85 * brightness, 1.0 * brightness, 1.0);
      } else if (colorType > 0.3) {
        colors.push(brightness, brightness, brightness, 1.0);
      } else {
        colors.push(1.0 * brightness, 0.95 * brightness, 0.8 * brightness, 1.0);
      }
    }
    const milkyWayStars = 4000;
    for (let i = 0; i < milkyWayStars; i++) {
      const theta = (Math.random() * 0.5 + 0.25) * Math.PI; 
      const phi = Math.random() * Math.PI * 2;
      const x = Math.sin(theta) * Math.cos(phi);
      const y = Math.sin(theta) * Math.sin(phi);
      const z = Math.cos(theta);
      stars.push(x, y, z);
      const brightness = 0.5 + Math.random() * 0.5;
      colors.push(0.85 * brightness, 0.9 * brightness, 1.0 * brightness, 1.0);
    }
    return {
      positions: new Float32Array(stars),
      colors: new Float32Array(colors),
      count: stars.length / 3,
    };
  }
  
  const starfield = createStarfieldSphere();
  
  const starPositionBuffer = device.createBuffer({ size: starfield.positions.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(starPositionBuffer, 0, starfield.positions);
  const starColorBuffer = device.createBuffer({ size: starfield.colors.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(starColorBuffer, 0, starfield.colors);

  const starfieldUniformBuffer = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

  const starfieldBindGroupLayout = device.createBindGroupLayout({
    entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: "uniform" } }],
  });

  const starfieldBindGroup = device.createBindGroup({
    layout: starfieldBindGroupLayout,
    entries: [{ binding: 0, resource: { buffer: starfieldUniformBuffer } }],
  });

  const starfieldShaderCode = `
  struct Uniforms { mvp: mat4x4<f32> };
  @group(0) @binding(0) var<uniform> uniforms: Uniforms;
  struct VertexOutput { @builtin(position) position: vec4<f32>, @location(0) color: vec4<f32> }
  @vertex
  fn vs_main(@location(0) pos: vec3<f32>, @location(1) color: vec4<f32>) -> VertexOutput {
    var output: VertexOutput;
    let starPos = pos * 50.0;
    var mvp = uniforms.mvp;
    mvp[3][0] = 0.0; mvp[3][1] = 0.0; mvp[3][2] = 0.0;
    let clipPos = mvp * vec4<f32>(starPos, 1.0);
    output.position = vec4<f32>(clipPos.xy, clipPos.w * 0.9999, clipPos.w);
    output.color = color;
    return output;
  }
  @fragment
  fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> { return input.color; }
  `;

  const starfieldShaderModule = device.createShaderModule({ code: starfieldShaderCode });

  const starfieldPipeline = device.createRenderPipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [starfieldBindGroupLayout] }),
    vertex: {
      module: starfieldShaderModule,
      entryPoint: "vs_main",
      buffers: [
        { arrayStride: 3 * 4, attributes: [{ shaderLocation: 0, offset: 0, format: "float32x3" }] },
        { arrayStride: 4 * 4, attributes: [{ shaderLocation: 1, offset: 0, format: "float32x4" }] },
      ],
    },
    fragment: {
      module: starfieldShaderModule,
      entryPoint: "fs_main",
      targets: [{ 
        format,
        blend: {
          color: { srcFactor: "src-alpha", dstFactor: "one", operation: "add" },
          alpha: { srcFactor: "one", dstFactor: "one", operation: "add" },
        },
      }],
    },
    primitive: { topology: "point-list" },
    depthStencil: { depthWriteEnabled: false, depthCompare: "less-equal", format: "depth24plus" },
  });

  const createTimeDisplay = () => {
    const overlay = document.createElement('div');
    overlay.id = 'time-overlay';
    overlay.style.cssText = `position:absolute;top:20px;right:20px;background:rgba(0,0,0,0.8);backdrop-filter:blur(10px);color:white;padding:16px;border-radius:8px;font-family:system-ui;font-size:14px;z-index:1000;min-width:280px;`;
    overlay.innerHTML = `
      <div style="font-size:18px;font-weight:600;margin-bottom:12px;">Real-Time Earth</div>
      <div id="time-display" style="margin-bottom:8px;"></div>
      <div id="local-time-display" style="font-size:12px;color:#aaa;margin-bottom:12px;"></div>
      <div id="sun-position" style="font-size:13px;color:#ffd700;"></div>
      <div style="margin-top:12px;padding-top:12px;border-top:1px solid #444;font-size:12px;color:#aaa;">
        • Drag to rotate<br>• Scroll to zoom
      </div>
    `;
    document.body.appendChild(overlay);
  };

  const updateTimeDisplay = (date: Date, sunLat: number, sunLon: number) => {
    const timeDisplay = document.getElementById('time-display');
    const localTimeDisplay = document.getElementById('local-time-display');
    const sunPosDisplay = document.getElementById('sun-position');
    if (timeDisplay) timeDisplay.textContent = `UTC: ${date.toUTCString()}`;
    if (localTimeDisplay) localTimeDisplay.textContent = `Local: ${date.toLocaleString()}`;
    if (sunPosDisplay) sunPosDisplay.textContent = `☀ Subsolar: ${sunLat.toFixed(2)}°N, ${sunLon.toFixed(2)}°E`;
  };

  createTimeDisplay();

  function calculateSunPosition(date: Date): { lat: number; lon: number; x: number; y: number; z: number } {
    const startOfYear = new Date(date.getFullYear(), 0, 0);
    const diff = date.getTime() - startOfYear.getTime();
    const dayOfYear = Math.floor(diff / 86400000);
    const declination = -23.45 * Math.cos((2 * Math.PI / 365) * (dayOfYear + 10));
    const declinationRad = declination * (Math.PI / 180);
    const hours = date.getUTCHours();
    const minutes = date.getUTCMinutes();
    const seconds = date.getUTCSeconds();
    const utcTime = hours + minutes / 60 + seconds / 3600;
    const solarLongitude = 180 - (utcTime * 15);
    const longitudeRad = solarLongitude * (Math.PI / 180);
    const x = -Math.cos(declinationRad) * Math.cos(longitudeRad);
    const y = -Math.sin(declinationRad);
    const z = -Math.cos(declinationRad) * Math.sin(longitudeRad);
    return { lat: declination, lon: solarLongitude, x, y, z };
  }

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

  const positionBuffer = device.createBuffer({ size: positions.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(positionBuffer, 0, positions);
  const uvBuffer = device.createBuffer({ size: uvs.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(uvBuffer, 0, uvs);
  const indexBuffer = device.createBuffer({ size: indices.byteLength, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(indexBuffer, 0, indices);

  const uniformBuffer = device.createBuffer({ size: 112, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: {} },
      { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
      { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: {} },
    ],
  });

  const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });

  let earthTexture = defaultTexture;
  let cloudTexture = defaultTexture;

  let bindGroup = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: earthTexture.createView() },
      { binding: 2, resource: sampler },
      { binding: 3, resource: cloudTexture.createView() },
    ],
  });

  async function loadEarthAssets() {
    try {
      earthTexture = await loadTexture("/earth_day.jpg");
    } catch(e) {}
    try {
      const date = new Date(Date.now() - 86400000).toISOString().split('T')[0];
      const url = `https://gibs.earthdata.nasa.gov/wmts/epsg4326/best/MODIS_Terra_CorrectedReflectance_TrueColor/default/${date}/250m/3/2/5.jpg`;
      const res = await fetch(url);
      if(res.ok) {
        const blob = await res.blob();
        const bmp = await createImageBitmap(blob);
        const tex = device.createTexture({ size: [bmp.width, bmp.height], format: 'rgba8unorm', usage: GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT});
        device.queue.copyExternalImageToTexture({source:bmp}, {texture:tex}, [bmp.width, bmp.height]);
        cloudTexture = tex;
      }
    } catch(e) { cloudTexture = createProceduralClouds(); }

    bindGroup = device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: earthTexture.createView() },
        { binding: 2, resource: sampler },
        { binding: 3, resource: cloudTexture.createView() },
      ],
    });
  }
  loadEarthAssets();

  function createProceduralClouds(): GPUTexture {
    const width = 2048; const height = 1024;
    const data = new Uint8Array(width * height * 4);
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const idx = (y * width + x) * 4;
        let value = Math.random() > 0.6 ? 100 : 0;
        data[idx] = value; data[idx + 1] = value; data[idx + 2] = value; data[idx + 3] = value * 2;
      }
    }
    const texture = device.createTexture({ size: [width, height], format: "rgba8unorm", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
    device.queue.writeTexture({ texture }, data, { bytesPerRow: width * 4 }, [width, height]);
    return texture;
  }

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

  struct VertexOutput { @builtin(position) position: vec4<f32>, @location(0) worldPos: vec3<f32>, @location(1) uv: vec2<f32> }

  @vertex
  fn vs_main(@location(0) position: vec3<f32>, @location(1) uv: vec2<f32>) -> VertexOutput {
    var output: VertexOutput;
    output.worldPos = position;
    output.uv = uv;
    output.position = uniforms.mvp * vec4<f32>(position, 1.0);
    return output;
  }

  @fragment
  fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let texColor = textureSample(earthTexture, textureSampler, input.uv);
    let cloudColor = textureSample(cloudTexture, textureSampler, input.uv);
    let normal = normalize(input.worldPos);
    let sunDir = normalize(uniforms.sunDir);
    let viewDir = normalize(uniforms.cameraPos - input.worldPos);
    let sunDot = dot(normal, sunDir);
    let terminatorWidth = 0.18;
    let daylight = smoothstep(-terminatorWidth, terminatorWidth, sunDot);
    let atmosphere = max(0.0, dot(normal, viewDir));
    let atmosphereGlow = pow(atmosphere, 5.0) * 0.12;
    let rim = 1.0 - max(0.0, dot(viewDir, normal));
    let rimGlow = pow(rim, 5.0) * 0.2 * max(0.0, sunDot + 0.15);
    let nightSide = 1.0 - daylight;
    let cityLights = nightSide * 0.25 * (texColor.r * 0.3 + texColor.g * 0.4 + texColor.b * 0.3);
    let visibleCityLights = cityLights * (1.0 - cloudColor.a * 0.5 * uniforms.cloudOpacity);
    let halfVec = normalize(sunDir + viewDir);
    let specular = pow(max(0.0, dot(normal, halfVec)), 80.0) * 0.25 * daylight * (1.0 - cloudColor.a * 0.6 * uniforms.cloudOpacity);
    let ambient = 0.25;
    let earthDiffuse = ambient + (1.0 - ambient) * daylight * 0.75;
    var earthColor = texColor.rgb * earthDiffuse;
    earthColor += vec3<f32>(1.0, 0.85, 0.5) * visibleCityLights;
    earthColor += vec3<f32>(0.85, 0.9, 1.0) * specular;
    let dynamicCloudAlpha = cloudColor.a * uniforms.cloudOpacity * 0.5;
    var finalColor = mix(earthColor, vec3<f32>(0.92), dynamicCloudAlpha);
    finalColor += vec3<f32>(0.25, 0.4, 0.7) * atmosphereGlow;
    finalColor += vec3<f32>(0.35, 0.5, 0.8) * rimGlow;
    return vec4<f32>(finalColor, 1.0);
  }
  `;

  const shaderModule = device.createShaderModule({ code: shaderCode });

  const pipeline = device.createRenderPipeline({
    layout: pipelineLayout,
    vertex: {
      module: shaderModule,
      entryPoint: "vs_main",
      buffers: [
        { arrayStride: 3 * 4, attributes: [{ shaderLocation: 0, offset: 0, format: "float32x3" }] },
        { arrayStride: 2 * 4, attributes: [{ shaderLocation: 1, offset: 0, format: "float32x2" }] },
      ],
    },
    fragment: { module: shaderModule, entryPoint: "fs_main", targets: [{ format }] },
    primitive: { topology: "triangle-list", cullMode: "back" },
    depthStencil: { depthWriteEnabled: true, depthCompare: "less", format: "depth24plus" },
  });

  // =================================================================
  // 6. RENDER LOOP
  // =================================================================
  let depthTexture: GPUTexture;
  function resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const width = Math.min(Math.floor(canvas.clientWidth * dpr), maxTextureSize);
    const height = Math.min(Math.floor(canvas.clientHeight * dpr), maxTextureSize);
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
      context.configure({ device, format, alphaMode: "opaque" });
      depthTexture = device.createTexture({ size: [width, height], format: "depth24plus", usage: GPUTextureUsage.RENDER_ATTACHMENT });
    }
  }
  window.addEventListener("resize", resize);
  resize();

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

  function multiplyMat4(a: Float32Array, b: Float32Array): Float32Array {
    const out = new Float32Array(16);
    for (let i = 0; i < 4; i++) {
      for (let j = 0; j < 4; j++) {
        out[i + j * 4] = a[i + 0 * 4] * b[0 + j * 4] + a[i + 1 * 4] * b[1 + j * 4] + a[i + 2 * 4] * b[2 + j * 4] + a[i + 3 * 4] * b[3 + j * 4];
      }
    }
    return out;
  }
  function rotationXMatrix(angle: number): Float32Array {
    const c = Math.cos(angle);
    const s = Math.sin(angle);
    return new Float32Array([1, 0, 0, 0, 0, c, s, 0, 0, -s, c, 0, 0, 0, 0, 1]);
  }
  function rotationYMatrix(angle: number): Float32Array {
    const c = Math.cos(angle);
    const s = Math.sin(angle);
    return new Float32Array([c, 0, -s, 0, 0, 1, 0, 0, s, 0, c, 0, 0, 0, 0, 1]);
  }
  function translationMatrix(x: number, y: number, z: number): Float32Array {
    return new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, x, y, z, 1]);
  }
  function getMVPMatrix() {
    if (canvas.width === 0 || canvas.height === 0) return new Float32Array(16);
    const aspect = canvas.width / canvas.height;
    const fov = Math.PI / 4;
    const near = 0.1;
    const far = 100;
    const f = 1 / Math.tan(fov / 2);
    const proj = new Float32Array([f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, (far + near) / (near - far), -1, 0, 0, (2 * far * near) / (near - far), 0]);
    const rotY = rotationYMatrix(rotationY);
    const rotX = rotationXMatrix(rotationX);
    const translate = translationMatrix(0, 0, -cameraDistance);
    const rotation = multiplyMat4(rotX, rotY);
    const view = multiplyMat4(translate, rotation);
    return multiplyMat4(proj, view);
  }

  canvas.addEventListener("mousedown", (e) => { isDragging = true; lastInteractionTime = Date.now(); lastMouseX = e.clientX; lastMouseY = e.clientY; canvas.style.cursor = "grabbing"; });
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
  canvas.addEventListener("mouseup", () => { isDragging = false; canvas.style.cursor = "grab"; });
  canvas.addEventListener("mouseleave", () => { isDragging = false; canvas.style.cursor = "grab"; });
  canvas.addEventListener("wheel", (e) => { e.preventDefault(); lastInteractionTime = Date.now(); const zoomSpeed = 0.3; cameraDistance += e.deltaY * zoomSpeed * 0.01; cameraDistance = Math.max(1.5, Math.min(10.0, cameraDistance)); });
  function getTouchDistance(touches: TouchList): number {
    if (touches.length < 2) return 0;
    const dx = touches[0].clientX - touches[1].clientX;
    const dy = touches[0].clientY - touches[1].clientY;
    return Math.sqrt(dx * dx + dy * dy);
  }
  canvas.addEventListener("touchstart", (e) => { e.preventDefault(); lastInteractionTime = Date.now(); if (e.touches.length === 1) { isDragging = true; lastMouseX = e.touches[0].clientX; lastMouseY = e.touches[0].clientY; } else if (e.touches.length === 2) { isPinching = true; lastTouchDistance = getTouchDistance(e.touches); } });
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
  canvas.addEventListener("touchend", (e) => { e.preventDefault(); if (e.touches.length === 0) { isDragging = false; isPinching = false; } else if (e.touches.length === 1) { isPinching = false; isDragging = true; lastMouseX = e.touches[0].clientX; lastMouseY = e.touches[0].clientY; } });
  canvas.style.cursor = "grab";

  function frame() {
    if (!depthTexture) { requestAnimationFrame(frame); return; }
    const timeSinceInteraction = Date.now() - lastInteractionTime;
    if (timeSinceInteraction > autoRotateDelay && !isDragging && !isPinching) { rotationY += autoRotateSpeed; }
    const now = new Date();
    const sunPos = calculateSunPosition(now);
    updateTimeDisplay(now, sunPos.lat, sunPos.lon);
    const mvpMatrix = getMVPMatrix();
    const camX = 0;
    const camY = 0;
    const camZ = cameraDistance;
    const cloudOpacity = Math.min(1.0, Math.max(0.0, (cameraDistance - 1.5) / 2.0));
    
    // Update Earth Uniforms
    const uniformData = new Float32Array(28);
    uniformData.set(mvpMatrix, 0);
    uniformData.set([sunPos.x, sunPos.y, sunPos.z], 16);
    uniformData.set([camX, camY, camZ], 20);
    uniformData[24] = cloudOpacity;
    device.queue.writeBuffer(uniformBuffer, 0, uniformData);

    // Update Skybox Uniforms
    device.queue.writeBuffer(skyboxUniformBuffer, 0, mvpMatrix);
    // Update Starfield Uniforms
    device.queue.writeBuffer(starfieldUniformBuffer, 0, mvpMatrix);

    const commandEncoder = device.createCommandEncoder();
    const textureView = context.getCurrentTexture().createView();
    const renderPass = commandEncoder.beginRenderPass({
      colorAttachments: [{ view: textureView, clearValue: { r: 0.0, g: 0.0, b: 0.02, a: 1 }, loadOp: "clear", storeOp: "store" }],
      depthStencilAttachment: { view: depthTexture.createView(), depthClearValue: 1.0, depthLoadOp: "clear", depthStoreOp: "store" },
    });

    // 1. Draw Skybox (Furthest Back)
    renderPass.setPipeline(skyboxPipeline);
    renderPass.setBindGroup(0, skyboxBindGroup);
    renderPass.setVertexBuffer(0, skyboxPosBuffer);
    renderPass.setVertexBuffer(1, skyboxUvBuffer);
    renderPass.setIndexBuffer(skyboxIdxBuffer, "uint32");
    renderPass.drawIndexed(skyboxGeo.count);

    // 2. Draw Stars (Middle)
    renderPass.setPipeline(starfieldPipeline);
    renderPass.setBindGroup(0, starfieldBindGroup);
    renderPass.setVertexBuffer(0, starPositionBuffer);
    renderPass.setVertexBuffer(1, starColorBuffer);
    renderPass.draw(starfield.count);

    // 3. Draw Earth (Front)
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