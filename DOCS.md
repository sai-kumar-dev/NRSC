## 🧠 Overall Concept: “Soft Real-Time Memory Governance”

This code is not trying to *optimize* memory.
It is trying to **prevent catastrophic failure**.

Think of it as a **thermostat + circuit breaker + emergency vent** for RAM and GPU memory.

These functions work together to:
• Cap memory usage
• Observe memory growth
• Actively shed memory when risk is detected

This is essential for **long-running scientific workloads** on large geospatial data.

---

## 🧩 The Three Roles (System View)

### 1️⃣ `set_memory_limits()` → **Hard Guardrail**

**Role:** Prevent the process from exceeding a safe memory ceiling.

What it enforces:
• OS-level virtual memory limit (`RLIMIT_AS`)
• Applies to heap + stack + mmap

Why this matters:
• Prevents silent memory ballooning
• Stops runaway allocations early
• Forces failures to be *controlled*, not catastrophic

This is **preventive engineering**.

---

### 2️⃣ `monitor_memory_usage()` → **Live Telemetry + Trigger**

**Role:** Continuously observe memory health during execution.

What it does:
• Reads RSS (real memory in RAM)
• Reads % of system memory used
• Logs for audit/debug
• Triggers cleanup if danger threshold crossed

Why this matters:
• Memory leaks are gradual
• Chunking still spikes memory temporarily
• Training loops amplify small leaks

This is **observability**, not control.

---

### 3️⃣ `clear_memory()` → **Emergency Vent**

**Role:** Actively release memory back to the system.

What it attempts:
• Python garbage collection
• GPU cache flushing (PyTorch)

Why this matters:
• Python doesn’t free memory eagerly
• CUDA allocators hoard memory
• Large tensors leave “ghost memory”

This is **damage control**, not optimization.

---

## 🔬 Why this is important in *your* pipeline

Your code does:
• xarray dataset loading
• netCDF memory mapping
• NumPy reshaping
• PyTorch tensor creation
• GPU training loops

Each of these:
• Allocates memory differently
• Releases memory differently
• Has different visibility to the OS

Without a memory subsystem:
➡️ Your program will work… until it doesn’t
➡️ Failures will be non-deterministic
➡️ Debugging becomes a nightmare

With this subsystem:
✔ Failures are earlier
✔ Logs explain *why*
✔ Runs become reproducible

---

## 🧠 Mental Model (Sticky Insight)

Imagine your program is a **city**:

• `set_memory_limits()` → city boundary wall
• `monitor_memory_usage()` → traffic cameras
• `clear_memory()` → emergency evacuation routes

You are not stopping traffic.
You are preventing **city collapse**.

---

## ⚠️ Important Reality Checks (very important)

### 1. `RLIMIT_AS` is OS-dependent

• Works reliably on Linux
• Spotty or ignored on Windows
• May not constrain GPU memory

So this is **best-effort**, not absolute law.

---

### 2. `gc.collect()` does NOT guarantee memory return to OS

Python often:
• Frees objects
• Keeps memory reserved

So:
✔ Prevents further growth
✖ Doesn’t always reduce RSS

That’s still useful.

---

### 3. `torch.cuda.empty_cache()` ≠ free GPU memory

It:
• Releases unused cached blocks
• Does NOT free tensors still referenced

Again: best-effort, not magic.

---

## 🧪 Why 75% threshold for cleanup?

Design logic:
• Cleanup before OS panic
• Leave headroom for spikes
• Avoid constant GC thrashing

Triggering at 90% is **too late**.
Triggering at 50% is **too aggressive**.

75% is a sane engineering compromise.

---

## 🧠 Final Takeaway

These functions do **not make your code faster**.
They make it **survivable, debuggable, and scalable**.

That’s the difference between:
• a notebook
• and a research pipeline


---

---



## 1️⃣ `create_ocean_mask(data_dict)`

### What this function *is* conceptually

This function answers one foundational question:

> “Which grid points in this massive spatio-temporal dataset actually represent the ocean?”

Everything else in the pipeline depends on this answer.

A **mask** is a boolean filter that decides:
• where learning is allowed
• where physics applies
• where interpolation is safe
• where NaNs must be preserved

---

### Why this function exists (the real problem)

Your dataset is a **lat × lon × time cube** that contains:
• ocean
• land
• coastlines
• missing values
• sensor gaps

PINNs **must not**:
• learn from land using ocean physics
• interpolate land temperatures as water
• hallucinate fluxes over continents

So you need a **physically grounded spatial filter**.

---

### Why SST is used as the primary indicator

Sea Surface Temperature is:
• present only over oceans
• physically bounded
• stable across datasets

The range:

```
271K to 308K  (≈ -2°C to 35°C)
```

is not arbitrary — it’s **ocean physics**:
• saltwater freezes below 0°C
• oceans don’t hit 60°C like deserts

This avoids:
• ice-covered land confusion
• desert false positives
• polar artifacts

---

### Why surface pressure (`sp`) is optionally included

This is a **secondary physical sanity check**.

Surface pressure over oceans:
• varies less abruptly
• stays within a known envelope
• behaves differently than over mountains

By intersecting SST-mask AND pressure-mask:
✔ coastlines get cleaner
✔ mountains get eliminated
✔ misclassified grid points drop

This is **data hygiene through physics**, not statistics.

---

### Why this function is critical

Without a correct ocean mask:
• your training data is corrupted
• physics constraints break silently
• results look “numerically fine” but physically wrong

This function is a **gatekeeper**.
Every downstream function assumes it did its job correctly.

---

## 2️⃣ `interpolate_nan_values(data, method='linear')`

### What this function *is* conceptually

This function repairs **missing measurements**, but in a **controlled, conservative way**.

It does **not** invent data.
It estimates data **only where justified**.

---

### Why interpolation is unavoidable here

Ocean datasets suffer from:
• cloud cover
• satellite dropouts
• sensor gaps
• coastal masking artifacts

If you don’t interpolate:
• tensors contain NaNs
• training crashes
• loss becomes NaN
• gradients explode

So interpolation is **mandatory**.

---

### Why interpolation is dangerous (and thus restricted)

Interpolation can:
• violate physics
• smear sharp fronts
• invent false gradients

So this function enforces **strict conditions**:

• If *all values are NaN* → abort
• If *too few valid points* → abort
• Only interpolate **locally in 2D**
• Fallback to nearest neighbor only when needed

This is **damage minimization**, not smoothing for beauty.

---

### Why `griddata` is used

`scipy.griddata`:
• works on irregular missing patterns
• supports linear + nearest
• respects spatial geometry

This is better than:
• blind convolution
• filling with mean
• global interpolation

The two-step approach:

1. Linear (smooth, physical)
2. Nearest (emergency fallback)

is a **robust interpolation strategy**.

---

### Why this function matters to PINNs

PINNs are extremely sensitive to:
• discontinuities
• NaNs
• artificial gradients

This function ensures:
✔ smooth inputs
✔ stable training
✔ physics terms don’t blow up

It is a **numerical stabilizer**.

---

## 3️⃣ `load_data(file_path)` — The Orchestrator

### What this function *really* is

This is not just a loader.

It is a **data validation + transformation + conditioning pipeline**.

Its job:

> “Turn raw NetCDF chaos into physics-ready tensors.”

---

### Why xarray is used

`xarray` is chosen because:
• NetCDF-native
• dimension-aware
• safer than raw NumPy
• metadata-friendly

This matters when handling:
• time
• latitude
• longitude
• variable alignment

---

### Why variable mapping exists

Real datasets are messy.

Variable names:
• change between sources
• change across versions
• are not standardized

By using `var_mapping`:
✔ code stays dataset-agnostic
✔ swapping datasets is easier
✔ logic stays clean

This is **defensive engineering**.

---

### Why data is extracted into `data_dict`

This does two things:
• decouples processing from xarray
• avoids lazy loading surprises

Once `.values` is called:
✔ you know memory cost
✔ you control lifetimes
✔ you can free memory later

This is essential for your memory strategy.

---

### Why ocean mask is created *before* interpolation

Order matters:

1. Identify valid ocean points
2. Interpolate **only there**
3. Preserve land NaNs forever

If you reverse this:
❌ land values get fabricated
❌ physics constraints collapse
❌ coastal zones get polluted

This order is **non-negotiable**.

---

### Why interpolation is done **per time step**

Because:
• spatial relationships change over time
• NaN patterns are time-dependent
• 3D interpolation is expensive and risky

So the design chooses:
✔ 2D spatial interpolation
✔ per-time-slice safety
✔ predictable memory usage

This is a **scalability decision**.

---

### Why land points are reset to NaN

This is a **hard constraint enforcement**.

Even if interpolation fills them:
→ they are forcefully removed

This guarantees:
✔ zero land leakage
✔ clean physics domain
✔ consistent masking

---

## 🧠 Big Picture Summary

Together, these functions:

• define the **physical domain**
• repair **measurement imperfections**
• protect **numerical stability**
• prepare data for **PINN training**

They are not “preprocessing”.
They are **scientific conditioning steps**.

---