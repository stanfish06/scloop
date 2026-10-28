module

public import Mathlib.Analysis.InnerProductSpace.PiL2
public import Mathlib.Data.Sym.Sym2
public import Mathlib.Data.ZMod.Basic
public import Mathlib.LinearAlgebra.Finsupp.LinearCombination
public import Mathlib.Tactic.NormNum.RealSqrt

/-!
# Image-equivalent loops and their representatives

`R` (reference) and `B` (bootstrap or second dataset) are vertex sets in a pseudometric space `V`.
Image persistence matches a cycle `z_R` of `Rips(R, t)` with a cycle `z_B` of `Rips(B, t)` when the
two are homologous in the union complex `Rips(R ∪ B, t)`. Scloop compare them inside `R`'s own
complex after pushing `z_B` along a map `π` that fixes `R`, sends `R ∪ B` into `R`, and moves
points by at most `δ`.

* `Homologous.image_proj`: image equivalence at scale `t` gives `z_R ~ π_# z_B` in
  `Rips(R, t + 2δ)`; `Homologous.image_proj_of_eq_zero` is the case `δ = 0` (no jitter).
* `TwoSquares`: an example where two classes are image equivalent but their loops are not in R.
  Two unit squares in `ℝ²` sharing an edge, plus one bootstrap point in the centre of the right
  square (boot+jit). The matched image class is nonzero on `[1, √2)`, the representatives are image
  equivalent there, but `z_R` and `π_# z_B` are not homologous in `Rips(R, s)` for any `s < √2`.
* `exists_not_homologous_iff`: the gap occurs iff `H₁(Rips(R, t)) → H₁(Rips(R ∪ B, t))` kills a
  class still alive in `Rips(R, s)`; `Homologous.kernel_dies`: such classes die by `t + 2δ`.
* `TightSquares`: `c` and `e` jittered towards each other by `δ`. Non-equivalence holds for every
  `s < t + 2δ`, so the constant `2` cannot be lowered (`TightSquares.not_equiv_below`).

Chains have coefficients in `ZMod 2`. Edges are `Sym2 V`, a 2-chain is a list of vertex triples, and
non-homology is certified by a 1-cocycle (`Homologous.eval_eq`).
-/

@[expose] public section

noncomputable section

open Finsupp

namespace Scloop.ImageMatching

section General

variable {V : Type*} [DecidableEq V]

/-- 1-chains over `𝔽₂`, indexed by unordered vertex pairs. -/
abbrev Chain1 (V : Type*) := Sym2 V →₀ ZMod 2

/-- The elementary chain of the edge `{a, b}`; a degenerate edge gives `0`. -/
def edge (a b : V) : Chain1 V := if a = b then 0 else single s(a, b) 1

/-- Boundary of the triangle `(a, b, c)`. -/
def bdry (τ : V × V × V) : Chain1 V := edge τ.1 τ.2.1 + edge τ.2.1 τ.2.2 + edge τ.1 τ.2.2

/-- Boundary of a 2-chain given as a list of triangles. -/
def bdry2 (c : List (V × V × V)) : Chain1 V := (c.map bdry).sum

/-- Boundary of a 1-chain. -/
def bdry1 : Chain1 V →ₗ[ZMod 2] V →₀ ZMod 2 :=
  linearCombination (ZMod 2) fun e ↦ Sym2.lift ⟨fun a b ↦ single a 1 + single b 1,
    fun _ _ ↦ add_comm _ _⟩ e

/-- `z` is a 1-cycle. -/
def IsCycle (z : Chain1 V) : Prop := bdry1 z = 0

/-- Pushforward of 1-chains along a vertex map; edges that collapse to a point are dropped. -/
def push (π : V → V) : Chain1 V →+ Chain1 V :=
  (filterAddHom fun e : Sym2 V ↦ ¬e.IsDiag).comp (mapDomain.addMonoidHom (Sym2.map π))

/-- Image of a triangle under a vertex map. -/
def mapTri (π : V → V) (τ : V × V × V) : V × V × V := (π τ.1, π τ.2.1, π τ.2.2)

theorem push_apply (π : V → V) (z : Chain1 V) :
    push π z = (mapDomain (Sym2.map π) z).filter fun e ↦ ¬e.IsDiag := rfl

theorem edge_apply (a b : V) (e : Sym2 V) :
    edge a b e = if a ≠ b ∧ s(a, b) = e then 1 else 0 := by
  unfold edge
  by_cases h : a = b <;> simp [h, single_apply]

theorem edge_self (a : V) : edge a a = 0 := by simp [edge]

theorem edge_comm (a b : V) : edge a b = edge b a := by
  unfold edge
  by_cases h : a = b
  · simp [h]
  · rw [ite_eq_right h, ite_eq_right (Ne.symm h), Sym2.eq_swap]

omit [DecidableEq V] in
theorem add_self_chain (z : Chain1 V) : z + z = 0 := by
  ext e
  simp only [Finsupp.add_apply, Finsupp.zero_apply]
  generalize z e = r
  revert r
  decide

omit [DecidableEq V] in
theorem neg_chain (z : Chain1 V) : -z = z := neg_eq_of_add_eq_zero_left (add_self_chain z)

/-- A triangle with a repeated vertex has zero boundary. -/
theorem bdry_degen (a b : V) : bdry (a, a, b) = 0 ∧ bdry (a, b, b) = 0 ∧ bdry (a, b, a) = 0 := by
  refine ⟨?_, ?_, ?_⟩ <;> simp only [bdry, edge_self, zero_add, add_zero]
  · exact add_self_chain _
  · exact add_self_chain _
  · rw [edge_comm b a, add_self_chain]

theorem push_edge (π : V → V) (a b : V) : push π (edge a b) = edge (π a) (π b) := by
  unfold edge
  by_cases hab : a = b
  · simp [hab]
  by_cases h : π a = π b
  · rw [ite_eq_right hab, ite_eq_left h, push_apply, mapDomain_single, Sym2.map_mk,
      filter_single_of_neg]
    simpa [Sym2.mk_isDiag_iff] using h
  · rw [ite_eq_right hab, ite_eq_right h, push_apply, mapDomain_single, Sym2.map_mk,
      filter_single_of_pos]
    simpa [Sym2.mk_isDiag_iff] using h

theorem push_bdry (π : V → V) (τ : V × V × V) : push π (bdry τ) = bdry (mapTri π τ) := by
  simp [bdry, mapTri, push_edge]

theorem push_bdry2 (π : V → V) (c : List (V × V × V)) :
    push π (bdry2 c) = bdry2 (c.map (mapTri π)) := by
  induction c with
  | nil => simp [bdry2]
  | cons τ c ih =>
    simp only [bdry2, List.map_cons, List.sum_cons, map_add] at ih ⊢
    rw [push_bdry, ih]

variable [PseudoMetricSpace V]

/-- `τ` is a triangle of the Vietoris–Rips complex `Rips(S, t)`. -/
def IsRipsTri (S : Set V) (t : ℝ) (τ : V × V × V) : Prop :=
  τ.1 ∈ S ∧ τ.2.1 ∈ S ∧ τ.2.2 ∈ S ∧
    dist τ.1 τ.2.1 ≤ t ∧ dist τ.2.1 τ.2.2 ≤ t ∧ dist τ.1 τ.2.2 ≤ t

/-- `z` and `z'` are homologous in `Rips(S, t)`: they differ by the boundary of a 2-chain. -/
def Homologous (S : Set V) (t : ℝ) (z z' : Chain1 V) : Prop :=
  ∃ c : List (V × V × V), (∀ τ ∈ c, IsRipsTri S t τ) ∧ z - z' = bdry2 c

/-- `z` is a 1-chain of `Rips(S, t)`: every edge in its support is a Rips edge. -/
def IsChain (S : Set V) (t : ℝ) (z : Chain1 V) : Prop :=
  ∀ e ∈ z.support, ∃ x y, e = s(x, y) ∧ x ≠ y ∧ x ∈ S ∧ y ∈ S ∧ dist x y ≤ t

/-- `φ` is a 1-cocycle of `Rips(S, t)`: it vanishes on every triangle boundary. -/
def IsCocycle (S : Set V) (t : ℝ) (φ : Chain1 V →+ ZMod 2) : Prop :=
  ∀ τ, IsRipsTri S t τ → φ (bdry τ) = 0

variable {S R B : Set V} {t δ : ℝ} {π : V → V}

theorem IsChain.add {z₁ z₂ : Chain1 V} (h₁ : IsChain S t z₁) (h₂ : IsChain S t z₂) :
    IsChain S t (z₁ + z₂) := by
  intro e he
  rcases Finset.mem_union.1 (support_add he) with h | h
  exacts [h₁ e h, h₂ e h]

theorem isChain_edge {a b : V} (ha : a ∈ S) (hb : b ∈ S) (hab : dist a b ≤ t) :
    IsChain S t (edge a b) := by
  intro e he
  unfold edge at he
  split_ifs at he with h
  · simp at he
  · obtain rfl := Finset.mem_singleton.1 (support_single_subset he)
    exact ⟨a, b, rfl, h, ha, hb, hab⟩

/-- A chain of `R` is fixed by any vertex map that fixes `R`. -/
theorem IsChain.push_eq_self (hfix : ∀ x ∈ S, π x = x) {z : Chain1 V} (hz : IsChain S t z) :
    push π z = z := by
  have hmap : mapDomain (Sym2.map π) z = z := by
    rw [mapDomain_congr (g := id), mapDomain_id]
    intro e he
    obtain ⟨x, y, rfl, -, hx, hy, -⟩ := hz e he
    simp [hfix x hx, hfix y hy]
  rw [push_apply, hmap, filter_eq_self_iff]
  intro e he
  obtain ⟨x, y, rfl, hxy, -⟩ := hz e (mem_support_iff.2 he)
  simpa [Sym2.mk_isDiag_iff] using hxy

omit [DecidableEq V] in
/-- `π` maps triangles of `Rips(R ∪ B, t)` to triangles of `Rips(R, t + 2δ)`. -/
theorem IsRipsTri.map (hmem : ∀ x ∈ R ∪ B, π x ∈ R) (hδ : ∀ x ∈ R ∪ B, dist x (π x) ≤ δ)
    {τ : V × V × V} (h : IsRipsTri (R ∪ B) t τ) : IsRipsTri R (t + 2 * δ) (mapTri π τ) := by
  obtain ⟨h₁, h₂, h₃, d₁₂, d₂₃, d₁₃⟩ := h
  have key : ∀ x ∈ R ∪ B, ∀ y ∈ R ∪ B, dist x y ≤ t → dist (π x) (π y) ≤ t + 2 * δ := by
    intro x hx y hy hxy
    have h4 := dist_triangle4 (π x) x y (π y)
    rw [dist_comm (π x) x] at h4
    linarith [hδ x hx, hδ y hy]
  exact ⟨hmem _ h₁, hmem _ h₂, hmem _ h₃, key _ h₁ _ h₂ d₁₂, key _ h₂ _ h₃ d₂₃, key _ h₁ _ h₃ d₁₃⟩

/-- Pushing along `π` turns a homology in `Rips(R ∪ B, t)` into one in `Rips(R, t + 2δ)`. -/
theorem Homologous.map (hmem : ∀ x ∈ R ∪ B, π x ∈ R) (hδ : ∀ x ∈ R ∪ B, dist x (π x) ≤ δ)
    {z z' : Chain1 V} (h : Homologous (R ∪ B) t z z') :
    Homologous R (t + 2 * δ) (push π z) (push π z') := by
  obtain ⟨c, hc, hzc⟩ := h
  refine ⟨c.map (mapTri π), ?_, ?_⟩
  · simp only [List.mem_map]
    rintro _ ⟨τ, hτ, rfl⟩
    exact (hc τ hτ).map hmem hδ
  · rw [← map_sub, hzc, push_bdry2]

/-- **Theorem.** If a chain `z_R` of `Rips(R, t)` is homologous to `z_B` in `Rips(R ∪ B, t)`, then
`z_R` is homologous to `π_# z_B` in `R`'s own complex `Rips(R, t + 2δ)`. -/
theorem Homologous.image_proj (hfix : ∀ x ∈ R, π x = x) (hmem : ∀ x ∈ R ∪ B, π x ∈ R)
    (hδ : ∀ x ∈ R ∪ B, dist x (π x) ≤ δ) {zR zB : Chain1 V} (hR : IsChain R t zR)
    (h : Homologous (R ∪ B) t zR zB) : Homologous R (t + 2 * δ) zR (push π zB) := by
  have := h.map hmem hδ
  rwa [hR.push_eq_self hfix] at this

/-- **Corollary (no jitter).** If `π` moves no point, image equivalence at scale `t` already gives
equivalence in `Rips(R, t)`. -/
theorem Homologous.image_proj_of_eq_zero (hfix : ∀ x ∈ R, π x = x)
    (hmem : ∀ x ∈ R ∪ B, π x ∈ R) (hδ : ∀ x ∈ R ∪ B, dist x (π x) = 0) {zR zB : Chain1 V}
    (hR : IsChain R t zR) (h : Homologous (R ∪ B) t zR zB) : Homologous R t zR (push π zB) := by
  simpa using h.image_proj hfix hmem (fun x hx ↦ (hδ x hx).le) hR

theorem homologous_iff_sub {z z' : Chain1 V} :
    Homologous S t z z' ↔ Homologous S t (z - z') 0 := by
  simp only [Homologous, sub_zero]

/-- A chain of `Rips(R, t)` that bounds in `Rips(R ∪ B, t)` already bounds in `Rips(R, t + 2δ)`:
classes killed by `B` die in `R` within `2δ`. -/
theorem Homologous.kernel_dies (hfix : ∀ x ∈ R, π x = x) (hmem : ∀ x ∈ R ∪ B, π x ∈ R)
    (hδ : ∀ x ∈ R ∪ B, dist x (π x) ≤ δ) {k : Chain1 V} (hk : IsChain R t k)
    (h : Homologous (R ∪ B) t k 0) : Homologous R (t + 2 * δ) k 0 := by
  simpa using h.image_proj hfix hmem hδ hk

/-- **Corollary (when the gap occurs).** For a chain `zR` of `Rips(R, t)`, some chain `zB` of
`Rips(R, t)` is image equivalent to `zR` at `t` but not equivalent to it in `Rips(R, s)` iff some
chain `k` of `Rips(R, t)` bounds in `Rips(R ∪ B, t)` but not in `Rips(R, s)`, i.e. the map
`H₁(Rips(R, t)) → H₁(Rips(R ∪ B, t))` kills a class still alive in `Rips(R, s)`. A map `π` fixing
`R` fixes chains of `Rips(R, t)`, so `zB = π_# zB` here. -/
theorem exists_not_homologous_iff {s : ℝ} {zR : Chain1 V} (hR : IsChain R t zR) :
    (∃ zB, IsChain R t zB ∧ Homologous (R ∪ B) t zR zB ∧ ¬Homologous R s zR zB) ↔
      ∃ k, IsChain R t k ∧ Homologous (R ∪ B) t k 0 ∧ ¬Homologous R s k 0 := by
  constructor
  · rintro ⟨zB, hB, h, hn⟩
    refine ⟨zR - zB, ?_, homologous_iff_sub.1 h, fun h' ↦ hn (homologous_iff_sub.2 h')⟩
    rw [sub_eq_add_neg, neg_chain]
    exact hR.add hB
  · rintro ⟨k, hk, h, hn⟩
    refine ⟨zR - k, ?_, ?_, ?_⟩
    · rw [sub_eq_add_neg, neg_chain]
      exact hR.add hk
    · rwa [homologous_iff_sub, sub_sub_cancel]
    · rwa [homologous_iff_sub, sub_sub_cancel]

theorem IsCocycle.apply_bdry2 {φ : Chain1 V →+ ZMod 2} (hφ : IsCocycle S t φ) :
    ∀ c : List (V × V × V), (∀ τ ∈ c, IsRipsTri S t τ) → φ (bdry2 c) = 0
  | [], _ => by simp [bdry2]
  | τ :: c, hc => by
    have ih := hφ.apply_bdry2 c fun τ' h ↦ hc τ' (List.mem_cons_of_mem _ h)
    simp only [bdry2, List.map_cons, List.sum_cons, map_add] at ih ⊢
    rw [hφ τ (hc τ List.mem_cons_self), ih, add_zero]

/-- Cocycle certificate: homologous chains take the same value on every cocycle. -/
theorem Homologous.eval_eq {φ : Chain1 V →+ ZMod 2} (hφ : IsCocycle S t φ) {z z' : Chain1 V}
    (h : Homologous S t z z') : φ z = φ z' := by
  obtain ⟨c, hc, hzc⟩ := h
  rw [← sub_eq_zero, ← map_sub, hzc, hφ.apply_bdry2 c hc]

/-- Evaluation at the edge `{u, v}` is a cocycle of `Rips(S, t)` when no third vertex of `S` lies
within `t` of both endpoints, i.e. when the edge lies in no Rips triangle. -/
theorem isCocycle_apply {u v : V} (huv : u ≠ v)
    (h : ∀ w ∈ S, dist w u ≤ t → dist w v ≤ t → w = u ∨ w = v) :
    IsCocycle S t (applyAddHom s(u, v)) := by
  have hS : ∀ q ∈ S, ∀ a b, s(a, b) = s(u, v) → dist q a ≤ t → dist q b ≤ t → q = a ∨ q = b := by
    rintro q hq a b hab hqa hqb
    rcases Sym2.eq_iff.1 hab with ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩
    · exact h q hq hqa hqb
    · exact (h q hq hqb hqa).symm
  rintro ⟨x, y, w⟩ ⟨hx, hy, hw, hxy, hyw, hxw⟩
  by_cases hdeg : x = y ∨ y = w ∨ x = w
  · rcases hdeg with rfl | rfl | rfl
    · rw [(bdry_degen x w).1, map_zero]
    · rw [(bdry_degen x y).2.1, map_zero]
    · rw [(bdry_degen x y).2.2, map_zero]
  push Not at hdeg
  obtain ⟨hxy', hyw', hxw'⟩ := hdeg
  -- in a non-degenerate triangle, the third vertex rules out every edge being `{u, v}`
  have h₁ : s(x, y) ≠ s(u, v) := fun he ↦ by
    rcases hS w hw x y he (by rwa [dist_comm]) (by rwa [dist_comm]) with rfl | rfl
    exacts [hxw' rfl, hyw' rfl]
  have h₂ : s(y, w) ≠ s(u, v) := fun he ↦ by
    rcases hS x hx y w he hxy hxw with rfl | rfl
    exacts [hxy' rfl, hxw' rfl]
  have h₃ : s(x, w) ≠ s(u, v) := fun he ↦ by
    rcases hS y hy x w he (by rwa [dist_comm]) hyw with rfl | rfl
    exacts [hxy' rfl, hyw' rfl]
  simp [bdry, edge_apply, h₁, h₂, h₃]

end General

/-! ### Example: two unit squares sharing an edge -/

namespace TwoSquares

/-- `a b c` on the bottom row, `d e f` on the top row, `p` at the centre of the right square. -/
inductive Pt
  | a | b | c | d | e | f | p
  deriving DecidableEq

open Pt

/-- Coordinates in `ℝ²`. -/
def pos : Pt → EuclideanSpace ℝ (Fin 2)
  | a => !₂[0, 0]
  | b => !₂[1, 0]
  | c => !₂[2, 0]
  | d => !₂[0, 1]
  | e => !₂[1, 1]
  | f => !₂[2, 1]
  | p => !₂[3 / 2, 1 / 2]

instance : PseudoMetricSpace Pt := PseudoMetricSpace.induced pos inferInstance

/-- Reference points: the six corners. -/
def R : Set Pt := {x | x ≠ p}

/-- Bootstrap points: the corners except `c`, plus `p`, a copy of `c` moved by `√2 / 2`. -/
def B : Set Pt := {x | x ≠ c}

/-- Nearest-point map onto `R`; `p` goes to `c`. -/
def proj (x : Pt) : Pt := if x = p then c else x

/-- Reference representative: the outer hexagon, i.e. the sum of both squares. -/
def zR : Chain1 Pt := edge a b + edge b c + edge c f + edge f e + edge e d + edge d a

/-- Bootstrap representative: the left square. -/
def zB : Chain1 Pt := edge a b + edge b e + edge e d + edge d a

/-- The four union triangles around `p`; they fill the right square. -/
def fill : List (Pt × Pt × Pt) := [(p, b, c), (p, c, f), (p, f, e), (p, e, b)]

/-- Squared Euclidean distance between two points. -/
def sqd (x y : Pt) : ℝ := (pos x 0 - pos y 0) ^ 2 + (pos x 1 - pos y 1) ^ 2

theorem dist_eq_sqrt (x y : Pt) : dist x y = √(sqd x y) := by
  change dist (pos x) (pos y) = _
  rw [EuclideanSpace.dist_eq, Fin.sum_univ_two, Real.dist_eq, Real.dist_eq, sq_abs, sq_abs, sqd]

theorem dist_le_sqrt {x y : Pt} {k : ℝ} (h : sqd x y ≤ k) : dist x y ≤ √k :=
  (dist_eq_sqrt x y).trans_le (Real.sqrt_le_sqrt h)

theorem dist_le_one {x y : Pt} (h : sqd x y ≤ 1) : dist x y ≤ 1 :=
  (dist_le_sqrt h).trans_eq Real.sqrt_one

/-- Pairs at squared distance at least `2` are not joined in `Rips(·, s)` for `s < √2`. -/
theorem not_dist_le {x y : Pt} {s : ℝ} (hs : s < √2) (h : 2 ≤ sqd x y) : ¬dist x y ≤ s :=
  not_le.2 (hs.trans_le ((Real.sqrt_le_sqrt h).trans_eq (dist_eq_sqrt x y).symm))

theorem proj_fix : ∀ x ∈ R, proj x = x := by
  intro x hx
  cases x <;> simp_all [proj, R]

theorem proj_mem : ∀ x ∈ R ∪ B, proj x ∈ R := by
  intro x _
  cases x <;> simp [proj, R]

theorem dist_proj_le : ∀ x ∈ R ∪ B, dist x (proj x) ≤ √2 / 2 := by
  intro x _
  cases x
  case p =>
    refine (dist_le_sqrt (k := 1 / 2) (by norm_num [sqd, pos, proj])).trans ?_
    rw [Real.sqrt_le_iff, div_pow, Real.sq_sqrt (by norm_num)]
    exact ⟨by positivity, by norm_num⟩
  all_goals simp only [proj, reduceCtorEq, ↓reduceIte, dist_self]; positivity

theorem zR_isCycle : IsCycle zR := by
  unfold IsCycle
  ext x
  cases x <;> simp [bdry1, zR, edge, linearCombination_single] <;> decide

theorem zB_isCycle : IsCycle zB := by
  unfold IsCycle
  ext x
  cases x <;> simp [bdry1, zB, edge, linearCombination_single] <;> decide

theorem zR_isChain : IsChain R 1 zR := by
  unfold zR
  repeat' apply IsChain.add
  all_goals exact isChain_edge (by simp [R]) (by simp [R]) (dist_le_one (by norm_num [sqd, pos]))

theorem zB_isChain : IsChain B 1 zB := by
  unfold zB
  repeat' apply IsChain.add
  all_goals exact isChain_edge (by simp [B]) (by simp [B]) (dist_le_one (by norm_num [sqd, pos]))

/-- `zB` also uses only reference vertices, so `proj` fixes it. -/
theorem proj_zB : push proj zB = zB := by
  refine IsChain.push_eq_self (S := R) (t := 1) proj_fix ?_
  unfold zB
  repeat' apply IsChain.add
  all_goals exact isChain_edge (by simp [R]) (by simp [R]) (dist_le_one (by norm_num [sqd, pos]))

/-- Image equivalence on the whole bar: `zR ~ zB` in `Rips(R ∪ B, s)` for every `s ≥ 1`. -/
theorem image_equiv {s : ℝ} (hs : 1 ≤ s) : Homologous (R ∪ B) s zR zB := by
  refine ⟨fill, ?_, ?_⟩
  · intro τ hτ
    simp only [fill, List.mem_cons, List.not_mem_nil, or_false] at hτ
    rcases hτ with rfl | rfl | rfl | rfl <;>
      exact ⟨by simp [R, B], by simp [R, B], by simp [R, B],
        (dist_le_one (by norm_num [sqd, pos])).trans hs,
        (dist_le_one (by norm_num [sqd, pos])).trans hs,
        (dist_le_one (by norm_num [sqd, pos])).trans hs⟩
  · ext q
    induction q using Sym2.ind with
    | h x y =>
      cases x <;> cases y <;>
        simp [zR, zB, bdry2, fill, bdry, edge, Finsupp.single_apply] <;> decide

/-- The matched image class is nonzero on `[1, √2)`: `zB` is not a boundary in the union. -/
theorem zB_not_boundary {s : ℝ} (hs : s < √2) : ¬Homologous (R ∪ B) s zB 0 := by
  intro h
  have hφ : IsCocycle (R ∪ B) s (applyAddHom s(d, a)) := by
    refine isCocycle_apply (by decide) fun w _ hwd hwa ↦ ?_
    cases w
    case a => exact .inr rfl
    case d => exact .inl rfl
    case e => exact absurd hwa (not_dist_le hs (by norm_num [sqd, pos]))
    all_goals exact absurd hwd (not_dist_le hs (by norm_num [sqd, pos]))
  have := h.eval_eq hφ
  simp [zB, edge] at this

/-- **Sharpness.** The representatives are not equivalent in `R`'s own complex below `√2`. -/
theorem not_equiv {s : ℝ} (hs : s < √2) : ¬Homologous R s zR (push proj zB) := by
  intro h
  have hφ : IsCocycle R s (applyAddHom s(c, f)) := by
    refine isCocycle_apply (by decide) fun w hw hwc hwf ↦ ?_
    cases w
    case c => exact .inl rfl
    case f => exact .inr rfl
    case b => exact absurd hwf (not_dist_le hs (by norm_num [sqd, pos]))
    case p => exact absurd hw (by simp [R])
    all_goals exact absurd hwc (not_dist_le hs (by norm_num [sqd, pos]))
  have := h.eval_eq hφ
  rw [proj_zB] at this
  simp [zR, zB, edge] at this

/-- At `s = √2` the diagonal `b f` enters `Rips(R, s)` and the representatives become equivalent. -/
theorem equiv_sqrt_two : Homologous R √2 zR (push proj zB) := by
  rw [proj_zB]
  refine ⟨[(b, c, f), (b, f, e)], ?_, ?_⟩
  · intro τ hτ
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hτ
    rcases hτ with rfl | rfl <;>
      exact ⟨by simp [R], by simp [R], by simp [R], dist_le_sqrt (by norm_num [sqd, pos]),
        dist_le_sqrt (by norm_num [sqd, pos]), dist_le_sqrt (by norm_num [sqd, pos])⟩
  · ext q
    induction q using Sym2.ind with
    | h x y =>
      cases x <;> cases y <;>
        simp [zR, zB, bdry2, bdry, edge, Finsupp.single_apply] <;> decide

/-- The general theorem applied here: equivalence at `1 + 2δ` with `δ = √2 / 2`. -/
theorem equiv_by_theorem : Homologous R (1 + 2 * (√2 / 2)) zR (push proj zB) :=
  (image_equiv le_rfl).image_proj proj_fix proj_mem dist_proj_le zR_isChain

end TwoSquares

/-! ### Example: the bound `t + 2δ` is attained -/

namespace TightSquares

/-- The squares of `TwoSquares`; `c'` and `e'` are copies of `c` and `e` moved towards each other
along the diagonal `c e` by `δ = √2 / 10`. -/
inductive Pt
  | a | b | c | d | e | f | c' | e'
  deriving DecidableEq

open Pt

/-- Coordinates in `ℝ²`. -/
def pos : Pt → EuclideanSpace ℝ (Fin 2)
  | a => !₂[0, 0]
  | b => !₂[1, 0]
  | c => !₂[2, 0]
  | d => !₂[0, 1]
  | e => !₂[1, 1]
  | f => !₂[2, 1]
  | c' => !₂[19 / 10, 1 / 10]
  | e' => !₂[11 / 10, 9 / 10]

instance : PseudoMetricSpace Pt := PseudoMetricSpace.induced pos inferInstance

/-- Reference points: the six corners. -/
def R : Set Pt := {x | x ≠ c' ∧ x ≠ e'}

/-- Bootstrap points: the corners with `c` and `e` replaced by their jittered copies. -/
def B : Set Pt := {x | x ≠ c ∧ x ≠ e}

/-- Sends each copy back to the point it was sampled from. -/
def proj (x : Pt) : Pt := if x = c' then c else if x = e' then e else x

/-- Reference representative: the outer hexagon. -/
def zR : Chain1 Pt := edge a b + edge b c + edge c f + edge f e + edge e d + edge d a

/-- Bootstrap representative: the left square through the copy `e'`. -/
def zB : Chain1 Pt := edge a b + edge b e' + edge e' d + edge d a

/-- The left square of `R`. -/
def zL : Chain1 Pt := edge a b + edge b e + edge e d + edge d a

/-- Six union triangles at scale `t = |c' e'| = 4√2 / 5`. -/
def fill : List (Pt × Pt × Pt) :=
  [(b, c, c'), (c, f, c'), (b, c', e'), (c', f, e'), (e', f, e), (e, d, e')]

/-- Squared Euclidean distance between two points. -/
def sqd (x y : Pt) : ℝ := (pos x 0 - pos y 0) ^ 2 + (pos x 1 - pos y 1) ^ 2

theorem dist_eq_sqrt (x y : Pt) : dist x y = √(sqd x y) := by
  change dist (pos x) (pos y) = _
  rw [EuclideanSpace.dist_eq, Fin.sum_univ_two, Real.dist_eq, Real.dist_eq, sq_abs, sq_abs, sqd]

theorem dist_le_sqrt {x y : Pt} {k : ℝ} (h : sqd x y ≤ k) : dist x y ≤ √k :=
  (dist_eq_sqrt x y).trans_le (Real.sqrt_le_sqrt h)

theorem dist_le_mul_sqrt_two {x y : Pt} {r : ℝ} (hr : 0 ≤ r) (h : sqd x y ≤ r ^ 2 * 2) :
    dist x y ≤ r * √2 := by
  refine (dist_le_sqrt h).trans_eq ?_
  rw [Real.sqrt_mul (by positivity), Real.sqrt_sq hr]

/-- Pairs at squared distance at least `2` are not joined in `Rips(·, s)` for `s < √2`. -/
theorem not_dist_le {x y : Pt} {s : ℝ} (hs : s < √2) (h : 2 ≤ sqd x y) : ¬dist x y ≤ s :=
  not_le.2 (hs.trans_le ((Real.sqrt_le_sqrt h).trans_eq (dist_eq_sqrt x y).symm))

theorem proj_fix : ∀ x ∈ R, proj x = x := by
  intro x hx
  cases x <;> simp_all [proj, R]

theorem proj_mem : ∀ x ∈ R ∪ B, proj x ∈ R := by
  intro x _
  cases x <;> simp [proj, R]

theorem dist_proj_le : ∀ x ∈ R ∪ B, dist x (proj x) ≤ 1 / 10 * √2 := by
  intro x _
  cases x
  case c' => exact dist_le_mul_sqrt_two (by norm_num) (by norm_num [sqd, pos, proj])
  case e' => exact dist_le_mul_sqrt_two (by norm_num) (by norm_num [sqd, pos, proj])
  all_goals simp only [proj, reduceCtorEq, ↓reduceIte, dist_self]; positivity

theorem zR_isChain : IsChain R (4 / 5 * √2) zR := by
  unfold zR
  repeat' apply IsChain.add
  all_goals
    exact isChain_edge (by simp [R]) (by simp [R])
      (dist_le_mul_sqrt_two (by norm_num) (by norm_num [sqd, pos]))

theorem zB_isChain : IsChain B (4 / 5 * √2) zB := by
  unfold zB
  repeat' apply IsChain.add
  all_goals
    exact isChain_edge (by simp [B]) (by simp [B])
      (dist_le_mul_sqrt_two (by norm_num) (by norm_num [sqd, pos]))

theorem zB_isCycle : IsCycle zB := by
  unfold IsCycle
  ext x
  cases x <;> simp [bdry1, zB, edge, linearCombination_single] <;> decide

theorem proj_zB : push proj zB = zL := by
  simp [zB, zL, push_edge, proj]

/-- Image equivalence from `t = 4√2 / 5` on: `zR ~ zB` in `Rips(R ∪ B, s)`. -/
theorem image_equiv {s : ℝ} (hs : 4 / 5 * √2 ≤ s) : Homologous (R ∪ B) s zR zB := by
  refine ⟨fill, ?_, ?_⟩
  · intro τ hτ
    simp only [fill, List.mem_cons, List.not_mem_nil, or_false] at hτ
    rcases hτ with rfl | rfl | rfl | rfl | rfl | rfl <;>
      exact ⟨by simp [R, B], by simp [R, B], by simp [R, B],
        (dist_le_mul_sqrt_two (by norm_num) (by norm_num [sqd, pos])).trans hs,
        (dist_le_mul_sqrt_two (by norm_num) (by norm_num [sqd, pos])).trans hs,
        (dist_le_mul_sqrt_two (by norm_num) (by norm_num [sqd, pos])).trans hs⟩
  · ext q
    induction q using Sym2.ind with
    | h x y =>
      cases x <;> cases y <;>
        simp [zR, zB, bdry2, fill, bdry, edge, Finsupp.single_apply] <;> decide

/-- The matched image class is nonzero below `√2`: `zB` is not a boundary in the union. -/
theorem zB_not_boundary {s : ℝ} (hs : s < √2) : ¬Homologous (R ∪ B) s zB 0 := by
  intro h
  have hφ : IsCocycle (R ∪ B) s (applyAddHom s(d, a)) := by
    refine isCocycle_apply (by decide) fun w _ hwd hwa ↦ ?_
    cases w
    case a => exact .inr rfl
    case d => exact .inl rfl
    case e => exact absurd hwa (not_dist_le hs (by norm_num [sqd, pos]))
    case e' => exact absurd hwa (not_dist_le hs (by norm_num [sqd, pos]))
    all_goals exact absurd hwd (not_dist_le hs (by norm_num [sqd, pos]))
  have := h.eval_eq hφ
  simp [zB, edge] at this

/-- The representatives are not equivalent in `R`'s own complex below `√2 = t + 2δ`. -/
theorem not_equiv {s : ℝ} (hs : s < √2) : ¬Homologous R s zR (push proj zB) := by
  intro h
  have hφ : IsCocycle R s (applyAddHom s(c, f)) := by
    refine isCocycle_apply (by decide) fun w hw hwc hwf ↦ ?_
    cases w
    case c => exact .inl rfl
    case f => exact .inr rfl
    case b => exact absurd hwf (not_dist_le hs (by norm_num [sqd, pos]))
    case c' => exact absurd hw (by simp [R])
    case e' => exact absurd hw (by simp [R])
    all_goals exact absurd hwc (not_dist_le hs (by norm_num [sqd, pos]))
  have := h.eval_eq hφ
  rw [proj_zB] at this
  simp [zR, zL, edge] at this

/-- At `s = √2` the diagonal `b f` enters `Rips(R, s)` and the representatives become equivalent. -/
theorem equiv_sqrt_two : Homologous R √2 zR (push proj zB) := by
  rw [proj_zB]
  refine ⟨[(b, c, f), (b, f, e)], ?_, ?_⟩
  · intro τ hτ
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hτ
    rcases hτ with rfl | rfl <;>
      exact ⟨by simp [R], by simp [R], by simp [R], dist_le_sqrt (by norm_num [sqd, pos]),
        dist_le_sqrt (by norm_num [sqd, pos]), dist_le_sqrt (by norm_num [sqd, pos])⟩
  · ext q
    induction q using Sym2.ind with
    | h x y =>
      cases x <;> cases y <;>
        simp [zR, zL, bdry2, bdry, edge, Finsupp.single_apply] <;> decide

/-- The general theorem at `t = 4√2 / 5`, `δ = √2 / 10`. -/
theorem equiv_by_theorem : Homologous R (4 / 5 * √2 + 2 * (1 / 10 * √2)) zR (push proj zB) :=
  (image_equiv le_rfl).image_proj proj_fix proj_mem dist_proj_le zR_isChain

/-- The theorem's scale `t + 2δ` is exactly `√2`, where `equiv_sqrt_two` holds and below which
`not_equiv` holds. -/
theorem bound_eq : 4 / 5 * √2 + 2 * (1 / 10 * √2) = √2 := by ring

/-- **Sharpness.** No bound `t + Cδ` with `C < 2` holds: here `zR` and `π_# zB` are still not
equivalent at that scale. -/
theorem not_equiv_below {C : ℝ} (hC : C < 2) :
    ¬Homologous R (4 / 5 * √2 + C * (1 / 10 * √2)) zR (push proj zB) :=
  not_equiv (by nlinarith [mul_pos (sub_pos.2 hC) (Real.sqrt_pos.2 (two_pos : (0 : ℝ) < 2))])

end TightSquares

end Scloop.ImageMatching

end

end
