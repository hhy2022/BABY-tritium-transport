import gmsh


def generate_baby_upper_2d_mesh(
    fname="baby_2d.msh",
    show_gui=True,
    mesh_size=0.001,
):
    gmsh.initialize()
    gmsh.model.add("BABY_upper_2D_OpenMC_based")
    occ = gmsh.model.occ

    cm = 1e-2

    # Radial dimensions [m]
    r_axis = 0.0
    r_heater = 0.439 * cm
    r_cllif = 7.000 * cm
    r_inconel = 7.300 * cm

    # Axial thicknesses [m]
    t_base = 0.786 * cm
    t_alumina = 0.635 * cm
    t_he = 0.600 * cm
    t_inconel = 0.300 * cm
    t_heater_gap = 0.878 * cm
    t_cllif = (6.388 + 0.13022) * cm
    t_gap = 4.605 * cm
    t_cap = 1.422 * cm

    # Axial coordinates [m]
    y0 = 0.0
    y1 = y0 + t_base
    y2 = y1 + t_alumina
    y3 = y2 + t_he
    y4 = y3 + t_inconel
    y5 = y4 + t_cllif
    y6 = y5 + t_gap

    y_he_top = y3
    y_inconel_top = y4
    y_cllif_bottom = y4
    y_cllif_top = y5
    y_gap_top = y6
    y_heater_bottom = y4 + t_heater_gap

    # =========================================================
    # Create surfaces
    # =========================================================
    s_IV_bottom = occ.addRectangle(r_axis, y_he_top, 0, r_inconel, t_inconel)
    s_cllif_lower = occ.addRectangle(
        r_axis, y_cllif_bottom, 0, r_cllif, y_heater_bottom - y_cllif_bottom
    )
    s_cllif_upper = occ.addRectangle(
        r_heater, y_heater_bottom, 0, r_cllif - r_heater, y_cllif_top - y_heater_bottom
    )
    s_IV_wall = occ.addRectangle(
        r_cllif, y_inconel_top, 0, r_inconel - r_cllif, y_gap_top - y_inconel_top
    )
    s_IV_top = occ.addRectangle(r_heater, y_gap_top, 0, r_inconel - r_heater, t_cap)

    occ.synchronize()

    # Fragment to resolve shared boundaries between all surfaces
    occ.fragment(
        [
            (2, s_IV_bottom),
            (2, s_IV_wall),
            (2, s_IV_top),
            (2, s_cllif_lower),
            (2, s_cllif_upper),
        ],
        [],
    )
    occ.synchronize()

    # =========================================================
    # Surface tag audit
    # =========================================================
    # for dim, tag in gmsh.model.getEntities(2):
    #     xc, yc, _ = occ.getCenterOfMass(dim, tag)
    #     print(f"tag={tag}  xc={xc:.5f}  yc={yc:.5f}")
    # exit()

    # Hardcoded surface tags after fragment (verified via audit)
    INCONEL_TAGS = [1, 2, 4]  # IV_bottom, IV_wall, IV_top
    CLLIF_TAGS = [3, 5]  # cllif_lower, cllif_upper

    # =========================================================
    # Volume physical groups
    # =========================================================
    gmsh.model.addPhysicalGroup(2, INCONEL_TAGS, tag=1)
    gmsh.model.setPhysicalName(2, 1, "inconel625")

    gmsh.model.addPhysicalGroup(2, CLLIF_TAGS, tag=3)
    gmsh.model.setPhysicalName(2, 3, "cllif_natural")

    # Colors: inconel625 -> steel blue, cllif_natural -> amber
    for tag in INCONEL_TAGS:
        gmsh.model.setColor([(2, tag)], 100, 149, 237)
    for tag in CLLIF_TAGS:
        gmsh.model.setColor([(2, tag)], 255, 180, 50)

    # =========================================================
    # Boundary physical groups
    # =========================================================
    gmsh.model.addPhysicalGroup(1, [1], tag=31)
    gmsh.model.setPhysicalName(1, 31, "inconel_outer_bottom")
    gmsh.model.addPhysicalGroup(1, [2, 6], tag=32)
    gmsh.model.setPhysicalName(1, 32, "inconel_outer_side")
    gmsh.model.addPhysicalGroup(1, [12, 13], tag=33)
    gmsh.model.setPhysicalName(1, 33, "inconel_outer_top")
    gmsh.model.addPhysicalGroup(1, [17], tag=21)
    gmsh.model.setPhysicalName(1, 21, "left_symmetry_liquid")
    gmsh.model.addPhysicalGroup(1, [5], tag=22)
    gmsh.model.setPhysicalName(1, 22, "left_symmetry_inconel")
    gmsh.model.addPhysicalGroup(1, [11], tag=11)
    gmsh.model.setPhysicalName(1, 11, "top_cap_bc")
    gmsh.model.addPhysicalGroup(1, [8], tag=12)
    gmsh.model.setPhysicalName(1, 12, "gap_sidewall_bc")
    gmsh.model.addPhysicalGroup(1, [18], tag=13)
    gmsh.model.setPhysicalName(1, 13, "liquid_surface_bc")
    gmsh.model.addPhysicalGroup(1, [14], tag=14)
    gmsh.model.setPhysicalName(1, 14, "heater_cap_bc")
    gmsh.model.addPhysicalGroup(1, [16, 19], tag=16)
    gmsh.model.setPhysicalName(1, 16, "liquid_heater_interface_bc")

    # =========================================================
    # CLLiF/Inconel interface
    # =========================================================
    boundary_cllif = set(
        gmsh.model.getBoundary([(2, t) for t in CLLIF_TAGS], oriented=False)
    )
    boundary_inconel = set(
        gmsh.model.getBoundary([(2, t) for t in INCONEL_TAGS], oriented=False)
    )
    curve_tags_interface = [c[1] for c in boundary_cllif.intersection(boundary_inconel)]

    gmsh.model.addPhysicalGroup(1, curve_tags_interface, tag=99)
    gmsh.model.setPhysicalName(1, 99, "liquid_inconel_interface")

    occ.synchronize()

    # =========================================================
    # Mesh generation with interface refinement
    # =========================================================
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), mesh_size)

    # Refine mesh near the CLLiF/Inconel interface
    f_dist = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(f_dist, "CurvesList", curve_tags_interface)

    f_thresh = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(f_thresh, "InField", f_dist)
    gmsh.model.mesh.field.setNumber(
        f_thresh, "SizeMin", mesh_size / 5
    )  # fine near interface
    gmsh.model.mesh.field.setNumber(f_thresh, "SizeMax", mesh_size)  # coarse far away
    gmsh.model.mesh.field.setNumber(f_thresh, "DistMin", 0.002)  # refine within 2 mm
    gmsh.model.mesh.field.setNumber(f_thresh, "DistMax", 0.010)  # transition over 10 mm
    gmsh.model.mesh.field.setAsBackgroundMesh(f_thresh)

    gmsh.model.mesh.generate(2)
    gmsh.write(fname)

    if show_gui:
        gmsh.fltk.run()

    gmsh.finalize()


if __name__ == "__main__":
    generate_baby_upper_2d_mesh()
