module

public import Mathlib.Algebra.Order.BigOperators.Group.Finset
public import Mathlib.Basic.Real.Basic
public import Mathlib.Data.Finset.SymmDiff
public import Mathlib.Tactic

/-!
# Scloop matching coherence

- `scloop.computing.coherence.compute_coherence(F, H, A)`
  - `A(F \ H) / A(F ∪ H)`
  - Source filling `F`, a matching certificate `H` and triangle areas `A`.
  - Over GF(2) the target filling is `F ∆ H`, so the score is the area-weighted Jaccard
    similarity of `F` and `F ∆ H`.

## Targets

* `Scloop.jaccardDist_triangle`: `d(X, Z) ≤ d(X, Y) + d(Y, Z)` for nonnegative areas.
* `Scloop.coherence_eq_one_sub_jaccardDist`: `coherence w F H = 1 - d(F, F ∆ H)`.
  This is used to verify that the code properly connects with the area-weighted Jaccard.

Together with `jaccardDist_self`, `jaccardDist_comm` and `jaccardDist_nonneg`, `jaccardDist` is a
pseudometric on fillings. Distinct fillings can be at distance `0` only through zero-area
triangles.
-/

@[expose] public section

open Finset
open scoped symmDiff

namespace Scloop

variable {α : Type*} [DecidableEq α] (w : α → ℝ)

/-- Total area of a finite set of triangles. -/
def area (s : Finset α) : ℝ := ∑ i ∈ s, w i

/-- Area-weighted Jaccard distance; `0` when the union has zero area. -/
noncomputable def jaccardDist (X Y : Finset α) : ℝ := area w (X ∆ Y) / area w (X ∪ Y)

/-- `compute_coherence(F, H, A)`; the Python `None` route is the zero-area union. -/
noncomputable def coherence (F H : Finset α) : ℝ := area w (F \ H) / area w (F ∪ H)

variable {w}

omit [DecidableEq α] in
lemma area_nonneg (hw : ∀ i, 0 ≤ w i) (s : Finset α) : 0 ≤ area w s :=
  sum_nonneg fun i _ ↦ hw i

omit [DecidableEq α] in
lemma area_mono (hw : ∀ i, 0 ≤ w i) {s t : Finset α} (h : s ⊆ t) : area w s ≤ area w t :=
  sum_le_sum_of_subset_of_nonneg h fun i _ _ ↦ hw i

lemma area_union_of_disjoint {s t : Finset α} (h : Disjoint s t) :
    area w (s ∪ t) = area w s + area w t :=
  sum_union h

lemma area_union_add_area_inter (s t : Finset α) :
    area w (s ∪ t) + area w (s ∩ t) = area w s + area w t :=
  sum_union_inter

private lemma div_le_add_div_add {x y t : ℝ} (hx : 0 ≤ x) (hxy : x ≤ y) (ht : 0 ≤ t) :
    x / y ≤ (x + t) / (y + t) := by
  rcases (hx.trans hxy).eq_or_lt with hy | hy
  · rw [← hy, div_zero]
    exact div_nonneg (by linarith) (by linarith)
  · rw [div_le_div_iff₀ hy (by linarith)]
    nlinarith [mul_le_mul_of_nonneg_left hxy ht]

private lemma div_le_div_of_le_of_le {x y z : ℝ} (hx : 0 ≤ x) (hxy : x ≤ y) (hyz : y ≤ z) :
    x / z ≤ x / y := by
  rcases (hx.trans hxy).eq_or_lt with hy | hy
  · have : x = 0 := by linarith
    simp [this]
  · exact div_le_div_of_nonneg_left hx hy hyz

@[simp]
lemma jaccardDist_self (X : Finset α) : jaccardDist w X X = 0 := by
  simp [jaccardDist, area]

lemma jaccardDist_comm (X Y : Finset α) : jaccardDist w X Y = jaccardDist w Y X := by
  rw [jaccardDist, jaccardDist, symmDiff_comm, union_comm]

lemma jaccardDist_nonneg (hw : ∀ i, 0 ≤ w i) (X Y : Finset α) : 0 ≤ jaccardDist w X Y :=
  div_nonneg (area_nonneg hw _) (area_nonneg hw _)

lemma jaccardDist_le_one (hw : ∀ i, 0 ≤ w i) (X Y : Finset α) : jaccardDist w X Y ≤ 1 :=
  div_le_one_of_le₀ (area_mono hw symmDiff_le_sup) (area_nonneg hw _)

/-- Raise `d(X, Z)` to the denominator `A(X ∪ Y ∪ Z)` by adding `A(Y \ (X ∪ Z))` to both terms,
bound the numerator by `A(X ∆ Y) + A(Y ∆ Z)`, then shrink each denominator back. -/
theorem jaccardDist_triangle (hw : ∀ i, 0 ≤ w i) (X Y Z : Finset α) :
    jaccardDist w X Z ≤ jaccardDist w X Y + jaccardDist w Y Z := by
  have hA := area_nonneg hw
  have hU : area w (X ∪ Z) + area w (Y \ (X ∪ Z)) = area w (X ∪ Y ∪ Z) := by
    rw [← area_union_of_disjoint disjoint_sdiff, union_sdiff_self_eq_union, union_right_comm]
  have hnum : area w (X ∆ Z) + area w (Y \ (X ∪ Z)) ≤ area w (X ∆ Y) + area w (Y ∆ Z) := by
    have h1 : area w (X ∆ Z) ≤ area w (X ∆ Y ∪ Y ∆ Z) := area_mono hw (symmDiff_triangle X Y Z)
    have h2 : area w (Y \ (X ∪ Z)) ≤ area w (X ∆ Y ∩ Y ∆ Z) := by
      refine area_mono hw fun i hi ↦ ?_
      simp only [mem_sdiff, mem_union, mem_inter, mem_symmDiff] at hi ⊢
      tauto
    linarith [area_union_add_area_inter (w := w) (X ∆ Y) (Y ∆ Z)]
  have hXY : area w (X ∆ Y) / area w (X ∪ Y ∪ Z) ≤ jaccardDist w X Y :=
    div_le_div_of_le_of_le (hA _) (area_mono hw symmDiff_le_sup)
      (area_mono hw subset_union_left)
  have hYZ : area w (Y ∆ Z) / area w (X ∪ Y ∪ Z) ≤ jaccardDist w Y Z := by
    refine div_le_div_of_le_of_le (hA _) (area_mono hw symmDiff_le_sup) (area_mono hw ?_)
    intro i hi
    simp only [mem_union] at hi ⊢
    tauto
  calc jaccardDist w X Z
      ≤ (area w (X ∆ Z) + area w (Y \ (X ∪ Z))) / (area w (X ∪ Z) + area w (Y \ (X ∪ Z))) :=
        div_le_add_div_add (hA _) (area_mono hw symmDiff_le_sup) (hA _)
    _ ≤ (area w (X ∆ Y) + area w (Y ∆ Z)) / area w (X ∪ Y ∪ Z) := by
        rw [hU]
        exact div_le_div_of_nonneg_right hnum (hA _)
    _ = area w (X ∆ Y) / area w (X ∪ Y ∪ Z) + area w (Y ∆ Z) / area w (X ∪ Y ∪ Z) :=
        add_div _ _ _
    _ ≤ jaccardDist w X Y + jaccardDist w Y Z := add_le_add hXY hYZ

/-- The score returned by `compute_coherence` is one minus the Jaccard distance between the source
filling `F` and the target filling `F ∆ H`. -/
theorem coherence_eq_one_sub_jaccardDist (F H : Finset α) (h : area w (F ∪ H) ≠ 0) :
    coherence w F H = 1 - jaccardDist w F (F ∆ H) := by
  have hU : F ∪ F ∆ H = F ∪ H := by
    ext i
    simp only [mem_union, mem_symmDiff]
    tauto
  have hsplit : area w (F \ H) + area w H = area w (F ∪ H) := by
    rw [← area_union_of_disjoint disjoint_sdiff_self_left, sdiff_union_self_eq_union]
  rw [coherence, jaccardDist, symmDiff_symmDiff_cancel_left, hU, eq_sub_iff_add_eq,
    ← add_div, hsplit, div_self h]

end Scloop
