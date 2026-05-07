import gmsh


def generate_baby_upper_2d_mesh(
    fname="baby_2d.msh",
    show_gui=True,
    mesh_size=0.001,
):
    gmsh.initialize()
    gmsh.model.add("BABY_upper_2D_OpenMC_based")
    occ = gmsh.model.occ
    # =========================================================
    # Unit conversion
    # =========================================================
    cm = 1e-2

    # =========================================================
    # Radial dimensions [m]
    # =========================================================
    r_axis = 0.0
    r_heater = 0.439 * cm
    r_cllif = 7.000 * cm
    r_inconel = 7.300 * cm

    # =========================================================
    # Axial thicknesses [m]
    # =========================================================
    t_base = 0.786 * cm
    t_alumina = 0.635 * cm
    t_he = 0.600 * cm
    t_inconel = 0.300 * cm
    t_heater_gap = 0.878 * cm
    t_cllif = (6.388 + 0.13022) * cm
    t_gap = 4.605 * cm
    t_cap = 1.422 * cm
    # =========================================================
    # Axial coordinates [cm]
    # =========================================================
    y0 = 0.0
    y1 = y0 + t_base
    y2 = y1 + t_alumina
    y3 = y2 + t_he
    y4 = y3 + t_inconel
    y5 = y4 + t_cllif
    y6 = y5 + t_gap
    y7 = y6 + t_cap

    y_he_top = y3
    y_inconel_top = y4
    y_cllif_bottom = y4
    y_cllif_top = y5
    y_gap_top = y6
    y_IV_top = y7

    y_heater_bottom = y4 + t_heater_gap
    # y_heater_top = y7  # heater ends at cap top

    # =========================================================
    # Create base surfaces
    # =========================================================
    surfaces = {}

    surfaces["IV_bottom"] = occ.addRectangle(r_axis, y_he_top, 0, r_inconel, t_inconel)

    # surfaces["heater"] = occ.addRectangle(
    #     r_axis, y_heater_bottom, 0, r_heater, y_heater_top - y_heater_bottom
    # )

    surfaces["cllif_lower"] = occ.addRectangle(
        r_axis, y_cllif_bottom, 0, r_cllif, y_heater_bottom - y_cllif_bottom
    )

    surfaces["cllif_upper"] = occ.addRectangle(
        r_heater, y_heater_bottom, 0, r_cllif - r_heater, y_cllif_top - y_heater_bottom
    )

    surfaces["IV_wall"] = occ.addRectangle(
        r_cllif, y_inconel_top, 0, r_inconel - r_cllif, y_gap_top - y_inconel_top
    )

    surfaces["IV_top"] = occ.addRectangle(
        r_heater, y_gap_top, 0, r_inconel - r_heater, t_cap
    )

    # surfaces["helium_box"] = occ.addRectangle(
    #     r_heater, y_cllif_top, 0, r_cllif - r_heater, y_gap_top - y_cllif_top
    # )

    occ.synchronize()

    all_objects = [
        (2, surfaces["IV_bottom"]),
        (2, surfaces["IV_wall"]),
        (2, surfaces["IV_top"]),
        (2, surfaces["cllif_lower"]),
        (2, surfaces["cllif_upper"]),
        # (2, surfaces["heater"]),
        # (2, surfaces["helium_box"]),
    ]
    occ.fragment(all_objects, [])
    occ.synchronize()

    # =========================================================
    # Reclassify surfaces
    # =========================================================
    physical_surfaces = {
        "inconel625": [],
        # "helium": [],
        "cllif_natural": [],
        # "heater": [],
    }

    tol = 1e-8

    def in_range(value, lower, upper):
        return (lower - tol) <= value <= (upper + tol)

    for dim, tag in gmsh.model.getEntities(2):
        xc, yc, _ = occ.getCenterOfMass(dim, tag)

        # Heater
        # if in_range(xc, r_axis, r_heater) and in_range(
        #     yc, y_heater_bottom, y_heater_top
        # ):
        #     physical_surfaces["heater"].append(tag)
        #     continue

        # CLLiF lower
        if in_range(yc, y_cllif_bottom, y_heater_bottom) and in_range(
            xc, r_axis, r_cllif
        ):
            physical_surfaces["cllif_natural"].append(tag)
            continue

        # CLLiF upper
        if in_range(yc, y_heater_bottom, y_cllif_top) and in_range(
            xc, r_heater, r_cllif
        ):
            physical_surfaces["cllif_natural"].append(tag)
            continue

        # Inconel bottom
        if in_range(yc, y_he_top, y_inconel_top) and in_range(xc, r_axis, r_inconel):
            physical_surfaces["inconel625"].append(tag)
            continue

        # Inconel wall
        if in_range(xc, r_cllif, r_inconel) and in_range(yc, y_inconel_top, y_gap_top):
            physical_surfaces["inconel625"].append(tag)
            continue

        # Inconel top
        if in_range(yc, y_gap_top, y_IV_top) and in_range(xc, r_heater, r_inconel):
            physical_surfaces["inconel625"].append(tag)
            continue

        # # Helium
        # if in_range(yc, y_cllif_top, y_gap_top) and in_range(xc, r_heater, r_cllif):
        #     physical_surfaces["helium"].append(tag)
        #     continue

    # Remove duplicates
    for key, tags in physical_surfaces.items():
        seen = set()
        deduped = []
        for tag in tags:
            if tag not in seen:
                seen.add(tag)
                deduped.append(tag)
        physical_surfaces[key] = deduped

    # =========================================================
    # Physical groups
    # =========================================================
    physical_ids = {
        "inconel625": 1,
        # "helium": 2,
        "cllif_natural": 3,
        # "heater": 4,
    }

    for name, group_id in physical_ids.items():
        tags = physical_surfaces[name]
        if tags:
            gmsh.model.addPhysicalGroup(2, tags, group_id)
            gmsh.model.setPhysicalName(2, group_id, name)

    # Color volume regions
    # inconel625 -> steel blue
    for tag in physical_surfaces["inconel625"]:
        gmsh.model.setColor([(2, tag)], 100, 149, 237)

    # cllif_natural -> amber
    for tag in physical_surfaces["cllif_natural"]:
        gmsh.model.setColor([(2, tag)], 255, 180, 50)

    # =========================================================
    # Boundary groups
    # =========================================================
    # outside_inconel = gmsh.model.addPhysicalGroup(1, [1, 2, 6, 12, 13], tag=10)
    # gmsh.model.setPhysicalName(1, outside_inconel, "outside_inconel")

    inconel_outer_bottom = gmsh.model.addPhysicalGroup(1, [1], tag=31)
    gmsh.model.setPhysicalName(1, inconel_outer_bottom, "inconel_outer_bottom")
    inconel_outer_side = gmsh.model.addPhysicalGroup(1, [2, 6], tag=32)
    gmsh.model.setPhysicalName(1, inconel_outer_side, "inconel_outer_side")
    inconel_outer_top = gmsh.model.addPhysicalGroup(1, [12, 13], tag=33)
    gmsh.model.setPhysicalName(1, inconel_outer_top, "inconel_outer_top")

    left_symmetry_liquid = gmsh.model.addPhysicalGroup(1, [17], tag=21)
    gmsh.model.setPhysicalName(1, left_symmetry_liquid, "left_symmetry_liquid")

    left_symmetry_inconel = gmsh.model.addPhysicalGroup(1, [5], tag=22)
    gmsh.model.setPhysicalName(1, left_symmetry_inconel, "left_symmetry_inconel")

    top_cap_bc = gmsh.model.addPhysicalGroup(1, [11], tag=11)
    gmsh.model.setPhysicalName(1, top_cap_bc, "top_cap_bc")

    gap_sidewall_bc = gmsh.model.addPhysicalGroup(1, [8], tag=12)
    gmsh.model.setPhysicalName(1, gap_sidewall_bc, "gap_sidewall_bc")

    liquid_surface_bc = gmsh.model.addPhysicalGroup(1, [18], tag=13)
    gmsh.model.setPhysicalName(1, liquid_surface_bc, "liquid_surface_bc")

    heater_cap_bc = gmsh.model.addPhysicalGroup(1, [14], tag=14)
    gmsh.model.setPhysicalName(1, heater_cap_bc, "heater_cap_bc")

    # heater_gap_bc = gmsh.model.addPhysicalGroup(1, [20], tag=15)
    # gmsh.model.setPhysicalName(1, heater_gap_bc, "heater_gap_bc")

    liquid_heater_interface_bc = gmsh.model.addPhysicalGroup(1, [16, 19], tag=16)
    gmsh.model.setPhysicalName(
        1, liquid_heater_interface_bc, "liquid_heater_interface_bc"
    )

    boundary_liquid = set(
        gmsh.model.getBoundary(
            [(2, tag) for tag in physical_surfaces["cllif_natural"]],
            oriented=False,
            recursive=False,
        )
    )
    boundary_inconel = set(
        gmsh.model.getBoundary(
            [(2, tag) for tag in physical_surfaces["inconel625"]],
            oriented=False,
            recursive=False,
        )
    )
    # boundary_heater = set(
    #     gmsh.model.getBoundary(
    #         [(2, tag) for tag in physical_surfaces["heater"]],
    #         oriented=False,
    #         recursive=False,
    #     )
    # )

    liquid_inconel_interface_curve = list(
        boundary_liquid.intersection(boundary_inconel)
    )
    # liquid_heater_interface_curve = list(boundary_liquid.intersection(boundary_heater))

    curve_tags_1 = [c[1] for c in liquid_inconel_interface_curve]
    # curve_tags_2 = [c[1] for c in liquid_heater_interface_curve]

    liquid_inconel_interface = gmsh.model.addPhysicalGroup(1, curve_tags_1, tag=99)
    gmsh.model.setPhysicalName(1, liquid_inconel_interface, "liquid_inconel_interface")

    # liquid_heater_interface = gmsh.model.addPhysicalGroup(1, curve_tags_2, tag=100)
    # gmsh.model.setPhysicalName(1, liquid_heater_interface, "liquid_heater_interface")
    gmsh.model.occ.synchronize()

    # =========================================================
    # Mesh
    # =========================================================
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), mesh_size)

    # Refine mesh near the CLLiF/Inconel interface
    f_dist_iface = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(f_dist_iface, "CurvesList", curve_tags_1)

    f_thresh_iface = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(f_thresh_iface, "InField", f_dist_iface)
    gmsh.model.mesh.field.setNumber(f_thresh_iface, "SizeMin", mesh_size / 5)
    gmsh.model.mesh.field.setNumber(f_thresh_iface, "SizeMax", mesh_size)
    gmsh.model.mesh.field.setNumber(f_thresh_iface, "DistMin", 0.002)
    gmsh.model.mesh.field.setNumber(f_thresh_iface, "DistMax", 0.006)

    # Refine mesh near the liquid surface
    f_dist_liquid = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(f_dist_liquid, "CurvesList", [18])

    f_thresh_liquid = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(f_thresh_liquid, "InField", f_dist_liquid)
    gmsh.model.mesh.field.setNumber(f_thresh_liquid, "SizeMin", mesh_size / 5)
    gmsh.model.mesh.field.setNumber(f_thresh_liquid, "SizeMax", mesh_size)
    gmsh.model.mesh.field.setNumber(f_thresh_liquid, "DistMin", 0.002)
    gmsh.model.mesh.field.setNumber(f_thresh_liquid, "DistMax", 0.006)

    # Use the finest size requested by either refinement field
    f_min = gmsh.model.mesh.field.add("Min")
    gmsh.model.mesh.field.setNumbers(
        f_min,
        "FieldsList",
        [f_thresh_iface, f_thresh_liquid],
    )

    gmsh.model.mesh.field.setAsBackgroundMesh(f_min)
    gmsh.model.mesh.generate(2)
    gmsh.write(fname)

    if show_gui:
        gmsh.fltk.run()

    gmsh.finalize()


if __name__ == "__main__":
    generate_baby_upper_2d_mesh()
