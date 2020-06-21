"""
Cube Solver
CFOP (Cross, F2L, OLL, PLL)
"""

from math import *

moves = []  # all moves containing both scramble and solution

# Euclidean coordinates, center of the cube as the origin
# U layer
a, b, c, d, e, f, g, h = (-1, -1, 1), (-1, 0, 1), (-1, 1, 1), (0, -1, 1), (0, 1, 1), (1, -1, 1), (1, 0, 1), (1, 1, 1)
# D layer
i, j, k, l, m, n, o, p = (1, -1, -1), (1, 0, -1), (1, 1, -1), (0, -1, -1), (0, 1, -1), (-1, -1, -1), (-1, 0, -1), (-1, 1, -1)
# M layer
q, r, s, t = (1, 1, 0), (1, -1, 0), (-1, -1, 0), (-1, 1, 0)
# center pieces (U, L, F, R, B, D)
u, v, w, x, y, z = (0, 0, 1), (0, -1, 0), (1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, 0, -1)


# Vector Representation of the cube, numbering each facet 1-48 excluding centers
# Order: orthogonal to [x, y, z] respectively, 0 if none
# Ex.: a: [35, 9, 1] where x-sticker is 35, y-sticker is 9, z-sticker is 1
# Reference group (initial cube, remains unchanged)
ref = {a: [35, 9, 1], b: [34, 0, 2], c: [33, 27, 3], d: [0, 10, 4], e: [0, 26, 5], f: [17, 11, 6], g: [18, 0, 7], h: [19, 25, 8],
       i: [22, 16, 41], j: [23, 0, 42], k: [24, 30, 43], l: [0, 15, 44], m: [0, 31, 45], n: [40, 14, 46], o: [39, 0, 47], p: [38, 32, 48],
       q: [21, 28, 0], r: [20, 13, 0], s: [37, 12, 0], t: [36, 29, 0],
       u: [0, 0, 'U'], v: [0, 'L', 0], w: ['F', 0, 0], x: [0, 'R', 0], y: ['B', 0, 0], z: [0, 0, 'D']}

# cube dict (changes as permutations occur)
cube = {a: [35, 9, 1], b: [34, 0, 2], c: [33, 27, 3], d: [0, 10, 4], e: [0, 26, 5], f: [17, 11, 6], g: [18, 0, 7], h: [19, 25, 8],
        i: [22, 16, 41], j: [23, 0, 42], k: [24, 30, 43], l: [0, 15, 44], m: [0, 31, 45], n: [40, 14, 46], o: [39, 0, 47], p: [38, 32, 48],
        q: [21, 28, 0], r: [20, 13, 0], s: [37, 12, 0], t: [36, 29, 0],
        u: [0, 0, 'U'], v: [0, 'L', 0], w: ['F', 0, 0], x: [0, 'R', 0], y: ['B', 0, 0], z: [0, 0, 'D']}

# a new cube dict that does not inherit characteristics of the original cube dict
new = cube.copy()


# relating layers to vectors: all facets (always on the same layer) got affected when performing a single basic move (rotation)
# Ex.: when doing U move, all eight facets initially on vectors (a, b, c, d, e, f, g, h) rotate 90 degrees cw about z-axis
layers = {"U": (a, b, c, d, e, f, g, h), "U'": (a, b, c, d, e, f, g, h),
          "D": (i, j, k, l, m, n, o, p), "D'": (i, j, k, l, m, n, o, p),
          "L": (a, d, f, s, r, n, l, i), "L'": (a, d, f, s, r, n, l, i),
          "F": (f, g, h, r, q, i, j, k), "F'": (f, g, h, r, q, i, j, k),
          "R": (h, e, c, q, t, k, m, p), "R'": (h, e, c, q, t, k, m, p),
          "B": (c, b, a, t, s, p, o, n), "B'": (c, b, a, t, s, p, o, n),
          "M": (b, u, g, w, j, z, o, y), "M'": (b, u, g, w, j, z, o, y)}


# Linear Transformation Representing Cube Rotation
# 90 degree rotation about x-axis clockwise
Tx = [[1, 0, 0],
      [0, 0, 1],
      [0, -1, 0]]
# 90 degree rotation about x-axis counter_clockwise
Tx_c = [[1, 0, 0],
        [0, 0, -1],
        [0, 1, 0]]
# 180 degree rotation about x-axis
Tx_2 = [[1, 0, 0],
        [0, -1, 0],
        [0, 0, -1]]
# 90 degree rotation about y-axis clockwise
Ty = [[0, 0, -1],
      [0, 1, 0],
      [1, 0, 0]]
# 90 degree rotation about y-axis counter_clockwise
Ty_c = [[0, 0, 1],
        [0, 1, 0],
        [-1, 0, 0]]
# 180 degree rotation about y-axis
Ty_2 = [[-1, 0, 0],
        [0, 1, 0],
        [0, 0, -1]]
# 90 degree rotation about z-axis clockwise
Tz = [[0, 1, 0],
      [-1, 0, 0],
      [0, 0, 1]]
# 90 degree rotation about z-axis counter_clockwise
Tz_c = [[0, -1, 0],
        [1, 0, 0],
        [0, 0, 1]]
# 180 degree rotation about z-axis
Tz_2 = [[-1, 0, 0],
        [0, -1, 0],
        [0, 0, 1]]


def mul(A, B):  # A is 3*3, B is 3*1
    """
    matrix multiplication
    :param A: 3*3 matrix (nested list)
    :param B: 3*1 matrix (tuple)
    :return: 3*1 matrix (tuple)
    """
    C = [0, 0, 0]
    for ni in range(3):
        for nj in range(3):
            C[ni] += A[ni][nj] * B[nj]
    return tuple(C)


def rot(T, V):
    """
    single rotation and its effect on individual facets
    :param T: matrix transformation
    :param V: a vector in the Euclidean space referring to a specific piece of the cube
    :return: updated new cube dict
    """
    for key in new:
        if key == mul(T, V):
            if T == Tx or T == Tx_c:  # rot about x-axis
                cube[V][1], cube[V][2] = cube[V][2], cube[V][1]  # y,z stickers swap, x-sticker stays unchanged
            elif T == Ty or T == Ty_c:  # rot about y-axis
                cube[V][0], cube[V][2] = cube[V][2], cube[V][0]  # x,z stickers swap, y-sticker stays unchanged
            elif T == Tz or T == Tz_c:  # rot about z-axis
                cube[V][0], cube[V][1] = cube[V][1], cube[V][0]  # x,y stickers swap, z-sticker stays unchanged
            new.update({key: cube[V]})  # move the facet from V to key


def basic(P):
    """
    basic permutations (single moves): when rotating the whole layer, all 8 facets on that layer got affected
    :param P: permutation (string)
    :return: updated cube dict
    """
    for V in layers[P]:
        if P == "U" or P == "D'":
            rot(Tz, V)
        elif P == "U'" or P == "D":
            rot(Tz_c, V)
        elif P == "F" or P == "B'":
            rot(Tx, V)
        elif P == "F'" or P == "B":
            rot(Tx_c, V)
        elif P == "R" or P == "L'" or P == "M'":
            rot(Ty, V)
        elif P == "R'" or P == "L" or P == "M":
            rot(Ty_c, V)
    cube.update(new)


# single turn
def U():
    basic("U")
    moves.append("U")
def Ui():
    """inverse of U"""
    basic("U'")
    moves.append("U'")
def F():
    basic("F")
    moves.append("F")
def Fi():
    """inverse of F"""
    basic("F'")
    moves.append("F'")
def R():
    basic("R")
    moves.append("R")
def Ri():
    """inverse of R"""
    basic("R'")
    moves.append("R'")
def D():
    basic("D")
    moves.append("D")
def Di():
    """inverse of D"""
    basic("D'")
    moves.append("D'")
def B():
    basic("B")
    moves.append("B")
def Bi():
    """inverse of B"""
    basic("B'")
    moves.append("B'")
def L():
    basic("L")
    moves.append("L")
def Li():
    """inverse of U"""
    basic("L'")
    moves.append("L'")
def M():
    basic("M")
    moves.append("M")
def Mi():
    basic("M'")
    moves.append("M'")


# double turns
def F2():
    F()
    F()
def U2():
    U()
    U()
def R2():
    R()
    R()
def B2():
    B()
    B()
def D2():
    D()
    D()
def L2():
    L()
    L()
def M2():
    M()
    M()


# common algorithms
def r_alg():
    """right alg"""
    R()
    U()
    Ri()
    Ui()
    
def l_alg():
    """left alg"""
    Li()
    Ui()
    L()
    U()
    
def li_alg():
    """inverse left alg"""
    U()
    L()
    Ui()
    Li()
    
def rb_alg():
    """backward right alg"""
    Ri()
    Ui()
    R()
    U()
    
def lb_alg():
    """backward left alg"""
    L()
    U()
    Li()
    Ui()
    
def f_alg():
    """front alg"""
    F()
    U()
    Fi()
    Ui()


# First Layer Cross
'''
it is usually called "white cross", but on can choose different faces to do a cross
done: J: [23, 0, 42]
      M: [0, 31, 45]
      O: [39, 0, 47]
      L: [0, 15, 44]
'''
# white edge stickers
white_edge = [42, 44, 45, 47]


def white_middle_matched():
    """
    bring all matched white pieces from the middle layer back to place (ei.first layer)
    till there is no matched white pieces in the middle layer
    note: to catch all the desirable cases at the same time, no elif or else statements are used
    """
    while True:

        if cube[r][0] != 23 and cube[q][0] != 23 and cube[q][1] != 31 and cube[t][1] != 31 and cube[r][1] != 15 and cube[s][1] != 15 and cube[s][0] != 39 and cube[t][0] != 39:
            break
        else:
            if cube[q][1] == 42:
                F()
            if cube[r][1] == 42:
                Fi()
            if cube[q][0] == 45:
                Ri()
            if cube[t][0] == 45:
                R()
            if cube[t][1] == 47:
                Bi()
            if cube[s][1] == 47:
                B()
            if cube[s][0] == 44:
                Li()
            if cube[r][0] == 44:
                L()


def white_top_oriented_matched():
    """
    bring all matched (color is matched with center) & oriented (white sticker on top) white pieces from the top layer back to place (ei.first layer)
    till there is no such white pieces on the top layer
    """
    if cube[g][2] == 42:
        F2()
    if cube[e][2] == 45:
        R2()
    if cube[b][2] == 47:
        B2()
    if cube[d][2] == 44:
        L2()


def white_top_oriented():
    """
    initial condition: white stick oriented (facing up) on top layer
    if necessary, rotate upper layer to align it with the correct center
    then do double turn to move it to the bottom
    """
    while True:
        if cube[g][2] not in white_edge and cube[e][2] not in white_edge and cube[b][2] not in white_edge and cube[d][2] not in white_edge:
            break
        else:
            white_top_oriented_matched()
            # white_top_oriented_unmatched
            if cube[g][2] == 44 or cube[e][2] == 42 or cube[b][2] == 45 or cube[d][2] == 47:
                U()
                white_top_oriented_matched()
            if cube[g][2] == 45 or cube[e][2] == 47 or cube[b][2] == 44 or cube[d][2] == 42:
                Ui()
                white_top_oriented_matched()
            if cube[g][2] == 47 or cube[e][2] == 44 or cube[b][2] == 42 or cube[d][2] == 45:
                U2()
                white_top_oriented_matched()


def white_top_disoriented():
    """
    initial condition: white stick disoriented (facing up) on top layer
    use algorithms to move it back to place
    """
    while True:
        if cube[g][0] not in white_edge and cube[e][1] not in white_edge and cube[b][0] not in white_edge and cube[d][1] not in white_edge:
            break
        else:
            if cube[g][0] == 42:
                Ui()
                Ri()
                F()
                R()
            if cube[g][0] == 45:
                F()
                Ri()
                Fi()
            if cube[g][0] == 47:
                Ui()
                R()
                Bi()
                Ri()
            if cube[g][0] == 44:
                Fi()
                L()
                F()
            if cube[e][1] == 42:
                Ri()
                F()
                R()
            if cube[e][1] == 45:
                U()
                F()
                Ri()
                Fi()
            if cube[e][1] == 47:
                R()
                Bi()
                Ri()
            if cube[e][1] == 44:
                U()
                Fi()
                L()
                F()
            if cube[b][0] == 42:
                U()
                Ri()
                F()
                R()
            if cube[b][0] == 45:
                Bi()
                R()
                B()
            if cube[b][0] == 47:
                U()
                R()
                Bi()
                Ri()
            if cube[b][0] == 44:
                B()
                Li()
                Bi()
            if cube[d][1] == 42:
                L()
                Fi()
                Li()
            if cube[d][1] == 45:
                U()
                Bi()
                R()
                B()
            if cube[d][1] == 47:
                Li()
                B()
                L()
            if cube[d][1] == 44:
                Ui()
                Fi()
                L()
                F()


def white_middle_unmatched():
    """
    initial condition: white piece in middle layer unmatched with the center piece
    bring it up to the top layer so that white piece faces up and then apply function white_top_oriented()
    """
    if cube[q][0] == 42 or cube[q][0] == 45 or cube[q][0] == 44 or cube[q][0] == 47:
        R()
        white_top_oriented()
    if cube[r][0] == 42 or cube[r][0] == 45 or cube[r][0] == 44 or cube[r][0] == 47:
        Li()
        white_top_oriented()
    if cube[t][0] == 42 or cube[t][0] == 45 or cube[t][0] == 44 or cube[t][0] == 47:
        Ri()
        white_top_oriented()
    if cube[s][0] == 42 or cube[s][0] == 45 or cube[s][0] == 44 or cube[s][0] == 47:
        L()
        white_top_oriented()
    if cube[q][1] == 42 or cube[q][1] == 45 or cube[q][1] == 44 or cube[q][1] == 47:
        Fi()
        white_top_oriented()
    if cube[r][1] == 42 or cube[r][1] == 45 or cube[r][1] == 44 or cube[r][1] == 47:
        F()
        white_top_oriented()
    if cube[t][1] == 42 or cube[t][1] == 45 or cube[t][1] == 44 or cube[t][1] == 47:
        B()
        white_top_oriented()
    if cube[s][1] == 42 or cube[s][1] == 45 or cube[s][1] == 44 or cube[s][1] == 47:
        Bi()
        white_top_oriented()


def white_bottom_disoriented():
    """
    initial condition: white piece on first layer disoriented
    dividing in to sevral cases and use different algorithms to bring it back to place
    """
    # white stick on j[0]
    if cube[j][0] == 44:
        F()
        L()
    if cube[j][0] == 45:
        Fi()
        Ri()
    if cube[j][0] == 42 or cube[j][0] == 47:
        F2()
        white_top_disoriented()
    # white stick on m[1]
    if cube[m][1] == 42:
        R()
        F()
    if cube[m][1] == 47:
        Ri()
        Bi()
    if cube[m][1] == 45 or cube[m][1] == 44:
        R2()
        white_top_disoriented()
    # white stick on o[0]
    if cube[o][0] == 45:
        B()
        R()
    if cube[o][0] == 44:
        Bi()
        Li()
    if cube[o][0] == 42 or cube[o][0] == 47:
        B2()
        white_top_disoriented()
    # white stick on m[1]
    if cube[l][1] == 47:
        L()
        B()
    if cube[l][1] == 42:
        Li()
        Fi()
    if cube[l][1] == 45 or cube[l][1] == 44:
        L2()
        white_top_disoriented()


def cross():
    """
    white cross, executing until cross solved
    """
    while True:
        if cube[j] == ref[j] and cube[m] == ref[m] and cube[o] == ref[o] and cube[l] == ref[l]:
            break
        else:
            white_middle_matched()
            white_top_oriented()
            white_top_disoriented()
            white_middle_unmatched()
            white_bottom_disoriented()


# 2-look F2L
# first layer
def corner(XY):
    """
    there are four kind of corners pieces in total in terms of its relative position to the faces
    consider a corner piece by the slot it lies on, ignore its orientation
    1. on the slot that intersects left and front layer (LF)
    2. on the slot that intersects right and back layer (RB)
    3. on the slot that intersects left and back layer (LB)
    4. on the slot that intersects right and front layer (RF)
    :param XY: slot
    """
    if XY == 'LF':
        while not cube[i] == ref[i]:
            l_alg()
    elif XY == 'RF':
        while not cube[k] == ref[k]:
            r_alg()
    elif XY == 'RB':
        while not cube[p] == ref[p]:
            rb_alg()
    elif XY == 'LB':
        while not cube[n] == ref[n]:
            lb_alg()


def basic_cases():
    """corner on slot with the same colors, on top or bottom layer"""
    if 41 in cube[f] or 41 in cube[i]:
        corner('LF')
    if 43 in cube[h] or 43 in cube[k]:
        corner('RF')
    if 48 in cube[c] or 48 in cube[p]:
        corner('RB')
    if 46 in cube[a] or 46 in cube[n]:
        corner('LB')


def top_corner_color_unmatched():
    """
    corner on top layer, unmatched with the slot in terms of colors
    """
    if 41 in cube[h] or 43 in cube[c] or 48 in cube[a] or 46 in cube[f]:
        U()
        basic_cases()
    if 41 in cube[c] or 43 in cube[a] or 48 in cube[f] or 46 in cube[h]:
        U2()
        basic_cases()
    if 41 in cube[a] or 43 in cube[f] or 48 in cube[h] or 46 in cube[c]:
        Ui()
        basic_cases()


def bottom_corner_color_unmatched():
    """
    corner on bottom layer, unmatched with the slot in terms of colors
    """
    if 43 in cube[i] or 48 in cube[i] or 46 in cube[i]:
        l_alg()
        basic_cases()
        top_corner_color_unmatched()
    if 41 in cube[k] or 48 in cube[k] or 46 in cube[k]:
        r_alg()
        basic_cases()
        top_corner_color_unmatched()
    if 41 in cube[p] or 43 in cube[p] or 46 in cube[p]:
        rb_alg()
        basic_cases()
        top_corner_color_unmatched()
    if 41 in cube[n] or 48 in cube[n] or 43 in cube[n]:
        lb_alg()
        basic_cases()
        top_corner_color_unmatched()


def first_layer():
    """
    keep executing algorithms (functions) above until first layer being solved
    """
    while True:
        if cube[i] == ref[i] and cube[k] == ref[k] and cube[p] == ref[p] and cube[n] == ref[n]:
            break
        else:
            basic_cases()
            top_corner_color_unmatched()
            bottom_corner_color_unmatched()


# second layer
def edge_top_matched():
    """
    edge piece on top, in a vector position where the sticker not facing top matched with the center piece in terms of color
    """
    if cube[g][0] == 20:
        Ui()
        Li()
        U()
        L()
        U()
        F()
        Ui()
        Fi()
    elif cube[g][0] == 21:
        U()
        R()
        Ui()
        Ri()
        Ui()
        Fi()
        U()
        F()
    if cube[e][1] == 28:
        Ui()
        Fi()
        U()
        F()
        U()
        R()
        Ui()
        Ri()
    elif cube[e][1] == 29:
        U()
        B()
        Ui()
        Bi()
        Ui()
        Ri()
        U()
        R()
    if cube[b][0] == 36:
        Ui()
        Ri()
        U()
        R()
        U()
        B()
        Ui()
        Bi()
    elif cube[b][0] == 37:
        U()
        L()
        Ui()
        Li()
        Ui()
        Bi()
        U()
        B()
    if cube[d][1] == 13:
        U()
        F()
        Ui()
        Fi()
        Ui()
        Li()
        U()
        L()
    elif cube[d][1] == 12:
        Ui()
        Bi()
        U()
        B()
        U()
        L()
        Ui()
        Li()


def edge_top_unmatched():
    """
    edge piece on top, in a vector position where the sticker not facing top has different color with the center piece under it
    """
    if cube[e][1] == 20 or cube[e][1] == 21 or cube[b][0] == 28 or cube[b][0] == 29 or cube[d][1] == 36 or cube[d][1] == 37 or cube[g][0] == 12 or cube[g][0] == 13:
        U()
        edge_top_matched()
    if cube[e][1] == 12 or cube[e][1] == 13 or cube[b][0] == 21 or cube[b][0] == 20 or cube[d][1] == 29 or cube[d][1] == 28 or cube[g][0] == 36 or cube[g][0] == 37:
        U2()
        edge_top_matched()
    if cube[e][1] == 36 or cube[e][1] == 37 or cube[b][0] == 12 or cube[b][0] == 13 or cube[d][1] == 20 or cube[d][1] == 21 or cube[g][0] == 28 or cube[g][0] == 29:
        Ui()
        edge_top_matched()


# edge piece in middle layer: all possible numbers on each facet except the solved one
wrong_q0 = [28, 12, 13, 20, 29, 36, 37]
wrong_t1 = [36, 12, 13, 20, 21, 28, 37]
wrong_s0 = [12, 13, 20, 21, 28, 29, 36]
wrong_r1 = [12, 20, 21, 28, 29, 36, 27]


def edge_middle():
    """
    edge piece in middle layer disoriented (ie. need to flip the piece so that the color matched)
    """
    if cube[q][0] in wrong_q0:
        R()
        Ui()
        Ri()
        Ui()
        Fi()
        U()
        F()
        edge_top_matched()
        edge_top_unmatched()
    if cube[t][1] in wrong_t1:
        B()
        Ui()
        Bi()
        Ui()
        Ri()
        U()
        R()
        edge_top_matched()
        edge_top_unmatched()
    if cube[s][0] in wrong_s0:
        L()
        Ui()
        Li()
        Ui()
        Bi()
        U()
        B()
        edge_top_matched()
        edge_top_unmatched()
    if cube[r][1] in wrong_r1:
        F()
        Ui()
        Fi()
        Ui()
        Li()
        U()
        L()
        edge_top_matched()
        edge_top_unmatched()


def second_layer():
    """
    keep executing the algorithms above until second layer being solved
    """
    while True:
        if cube[r] == ref[r] and cube[q] == ref[q] and cube[t] == ref[t] and cube[s] == ref[s]:
            break
        else:
            edge_top_matched()
            edge_top_unmatched()
            edge_middle()


# 2-look OLL
'''
stage1 - 3 cases:
Bar: F r_alg Fi
L shape (reflected L, tip pointing bottom right corner): B li_alg Bi
Dot: (F r_alg Fi) (B li_alg Bi) = Bar + L shape
'''

def bar():
    """horizontal bar"""
    F()
    r_alg()
    Fi()
    
def Lshape():
    """L pointing top left, ie. open toward bottom right"""
    B()
    li_alg()
    Bi()
    
def dot():
    """dot at center, maybe with other yellow stickers scatter on top layer but not bar or Lshape"""
    bar()
    Lshape()


# yellow edge stickers
side = [2, 5, 7, 4]


def oll1():
    """
    also called the yellow cross (as long as a yellow cross is there, regardless of the corners, it is solved)
    doing nothing if yellow cross already presented;
    otherwise apply the algorithms above to solve the yellow cross
    """
    if cube[d][2] in side and cube[e][2] in side and cube[b][2] in side and cube[g][2] in side:
        return True
    else:
        if cube[d][2] in side:
            if cube[e][2] in side:  # horizontal bar
                bar()
            elif cube[b][2] in side:  # L pointing bottom right
                U2()
                Lshape()
            elif cube[g][2] in side:  # L pointing top right
                Ui()
                Lshape()
        elif cube[b][2] in side:
            if cube[g][2] in side:  # vertical bar
                U()
                bar()
            elif cube[e][2] in side:  # L pointing bottom left
                U()
                Lshape()
        elif cube[e][2] in side and cube[g][2] in side:  # L pointing top left
            Lshape()
        elif cube[b][2] not in side and cube[e][2] not in side and cube[g][2] not in side and cube[d][2] not in side:
            dot()


'''
stage2 - 7 cases
missing 4 pieces
1. Hshape: F (r_alg r_alg r_alg) Fi
2. pi: R U2 (R2 Ui R2 Ui R2) U2 R
missing 3 pieces
3. left fish: R U Ri U R U2 Ri
4. right fish: Li Ui L Ui Li U2 L
missing 2 pieces
5. Ushape (pieces on front layer): R2 D Ri U2 R Di Ri U2 Ri
6. Tshape (pieces to the right): Ri Fi L F R Fi Li F
7. Bowtie (left and back): Ri Fi Li F R Fi L F
'''

def Hshape():
    F()
    r_alg()
    r_alg()
    r_alg()
    Fi()
    
def pi():
    R()
    U2()
    R2()
    Ui()
    R2()
    Ui()
    R2()
    U2()
    R()
    
def left_fish():
    # head of fish on left
    R()
    U()
    Ri()
    U()
    R()
    U2()
    Ri()
    
def right_fish():
    # head of fish on right
    Li()
    Ui()
    L()
    Ui()
    Li()
    U2()
    L()
    
def Ushape():
    R2()
    D()
    Ri()
    U2()
    R()
    Di()
    Ri()
    U2()
    Ri()
    
def Tshape():
    Ri()
    Fi()
    L()
    F()
    R()
    Fi()
    Li()
    F()
    
def bowtie():
    Ri()
    Fi()
    Li()
    F()
    R()
    Fi()
    L()
    F()


# yellow corner stickers
corners = [1, 3, 6, 8]


def oll2():
    """
    orient last layer (only the stickers facing up)
    need only one algorithm
    """
    if cube[a][2] in corners and cube[c][2] in corners and cube[h][2] in corners and cube[f][2] in corners:
        # already oriented, no need to go through this stage
        return True
    else:
        # case 1 - two stickers on the same face
        # F
        if cube[f][0] in corners and cube[h][0] in corners:
            if cube[a][0] in corners and cube[c][0] in corners:
                Hshape()
            elif cube[a][1] in corners and cube[c][1] in corners:
                U()
                pi()
            elif cube[a][0] not in corners and cube[c][0] not in corners and cube[a][1] not in corners and cube[c][1] not in corners:
                Ushape()
        # B
        elif cube[a][0] in corners and cube[c][0] in corners:
            if cube[f][1] in corners and cube[h][1] in corners:
                Ui()
                pi()
            elif cube[f][0] not in corners and cube[h][0] not in corners and cube[f][1] not in corners and cube[h][1] not in corners:
                U2()
                Ushape()
        # R
        elif cube[c][1] in corners and cube[h][1] in corners:
            if cube[a][1] in corners and cube[f][1] in corners:
                U()
                Hshape()
            elif cube[a][0] in corners and cube[f][0] in corners:
                U2()
                pi()
            elif cube[a][0] not in corners and cube[f][0] not in corners and cube[a][1] not in corners and cube[f][1] not in corners:
                U()
                Ushape()
        # L
        elif cube[a][1] in corners and cube[f][1] in corners:
            if cube[c][0] in corners and cube[h][0] in corners:
                pi()
            elif cube[c][0] not in corners and cube[h][0] not in corners and cube[c][1] not in corners and cube[h][1] not in corners:
                Ui()
                Ushape()
        # case2 - T shape
        # F (slot)
        elif cube[f][1] in corners and cube[h][1] in corners:
            Ui()
            Tshape()
        # B
        elif cube[a][1] in corners and cube[c][1] in corners:
            U()
            Tshape()
        # R
        elif cube[c][0] in corners and cube[h][0] in corners:
            Tshape()
        # L
        elif cube[a][0] in corners and cube[f][0] in corners:
            U2()
            Tshape()
        # case3 - Bowtie
        # 1
        elif cube[c][0] in corners and cube[f][1] in corners:
            bowtie()
        # 2
        elif cube[a][1] in corners and cube[h][0] in corners:
            U()
            bowtie()
        # 3
        elif cube[c][0] in corners and cube[f][1] in corners:
            U2()
            bowtie()
        # 4
        elif cube[a][0] in corners and cube[h][1] in corners:
            Ui()
            bowtie()
        # case4 - left and right fishes
        # bottom left
        elif cube[f][2] in corners:
            if cube[h][0] in corners:
                left_fish()
            elif cube[h][1] in corners:
                Ui()
                right_fish()
        # top left
        elif cube[a][2] in corners:
            if cube[f][1] in corners:
                Ui()
                left_fish()
            elif cube[f][0] in corners:
                U2()
                right_fish()
        # top right
        elif cube[c][2] in corners:
            if cube[a][0] in corners:
                U2()
                left_fish()
            elif cube[a][1] in corners:
                U()
                right_fish()
        # bottom right
        elif cube[h][2] in corners:
            if cube[c][1] in corners:
                U()
                left_fish()
            elif cube[c][0] in corners:
                right_fish()



# PLL
# 21 cases

# permutation of edges or corners only
def Ua():
    R()
    Ui()
    R()
    U()
    R()
    U()
    R()
    Ui()
    Ri()
    Ui()
    R2()
    
def Ub():
    R2()
    U()
    r_alg()
    Ri()
    Ui()
    Ri()
    U()
    Ri()
    
def Aa():
    '''square at bottom left'''
    Ri()
    F()
    Ri()
    B2()
    R()
    Fi()
    Ri()
    B2()
    R2()
    
def Ab():
    '''square at top left'''
    R()
    Bi()
    R()
    F2()
    Ri()
    B()
    R()
    F2()
    R2()
    
def Z():
    M2()
    U()
    M2()
    U()
    Mi()
    U2()
    M2()
    U2()
    Mi()
    U2()
    
def H():
    M2()
    U()
    M2()
    U2()
    M2()
    U()
    M2()
    
def E():
    R()
    Bi()
    Ri()
    F()
    R()
    B()
    Ri()
    Fi()
    R()
    B()
    Ri()
    F()
    R()
    Bi()
    Ri()
    Fi()
    
# swap one set of adjacent corners
def Ra():
    L()
    U2()
    Li()
    U2()
    L()
    Fi()
    l_alg()
    L()
    F()
    L2()
    U()
    
def Rb():
    Ri()
    U2()
    R()
    U2()
    Ri()
    F()
    r_alg()
    Ri()
    Fi()
    R2()
    Ui()
    
def Ja():
    Ri()
    U()
    Li()
    U2()
    R()
    Ui()
    Ri()
    U2()
    R()
    L()
    Ui()
    
def Jb():
    R()
    U()
    Ri()
    Fi()
    r_alg()
    Ri()
    F()
    R2()
    Ui()
    Ri()
    Ui()
    
def T():
    r_alg()
    Ri()
    F()
    R2()
    Ui()
    Ri()
    Ui()
    R()
    U()
    Ri()
    Fi()
    
def Fshape():
    Ri()
    U2()
    Ri()
    Ui()
    Bi()
    Ri()
    B2()
    Ui()
    Bi()
    U()
    Bi()
    R()
    B()
    Ui()
    R()
    
# swap one set of corners diagonally
def V():
    Ri()
    U()
    Ri()
    Ui()
    Bi()
    Ri()
    B2()
    Ui()
    Bi()
    U()
    Bi()
    R()
    B()
    R()
    
def Y():
    F()
    R()
    Ui()
    Ri()
    Ui()
    R()
    U()
    Ri()
    Fi()
    r_alg()
    Ri()
    F()
    R()
    Fi()
    
def Na():
    L()
    Ui()
    R()
    U2()
    Li()
    U()
    Ri()
    L()
    Ui()
    R()
    U2()
    Li()
    U()
    Ri()
    U()
    
def Nb():
    Ri()
    U()
    Li()
    U2()
    R()
    Ui()
    L()
    Ri()
    U()
    Li()
    U2()
    R()
    Ui()
    L()
    Ui()
    
# double spins
def Ga():
    R2()
    D()
    Bi()
    U()
    Bi()
    Ui()
    B()
    Di()
    R2()
    Fi()
    U()
    F()
    
def Gb():
    Ri()
    Ui()
    R()
    B2()
    D()
    Li()
    U()
    L()
    Ui()
    L()
    Di()
    B2()
    
def Gc():
    R2()
    Di()
    F()
    Ui()
    F()
    U()
    Fi()
    D()
    R2()
    B()
    Ui()
    Bi()
    
def Gd():
    R()
    U()
    Ri()
    F2()
    Di()
    L()
    Ui()
    Li()
    U()
    Li()
    D()
    F2()


def back_slot_completed():
    """
    intersecting slot of back and top layer completed
    possible cases: Ua, Ub, Ja, Jb, Fshape
    """
    if abs(cube[a][1] - cube[f][1]) == 2:
        if abs(cube[a][1] - cube[e][1]) == 1:
            Ua()
        else:
            Ub()
    elif abs(cube[a][1] - cube[d][1]) == 1:
        U2()
        Ja()
    elif abs(cube[c][1] - cube[e][1]) == 1:
        Ui()
        Jb()
    else:
        Fshape()


def bottom_left_square():
    """
    top layer has a completed 2 by 2 square at bottom left corner
    possible cases: Aa, Ab, V
    """
    if abs(cube[a][0] - cube[c][0]) == 2:
        Aa()
    elif abs(cube[c][1] - cube[h][1]) == 2:
        U()
        Ab()
    else:
        V()


def bottom_left_slot():
    """
    1*2 slot at bottom left
    possible cases: T, Ra, Y, Gd, Gc
    """
    if abs(cube[f][1] - cube[a][1]) == 2:
        if abs(cube[a][0] - cube[b][0]) == 1:
            T()
        else:
            Ui()
            Ra()
    elif abs(cube[c][1] - cube[e][1]) == 1:
        Y()
    elif abs(cube[a][0] - cube[c][0]) == 2:
        Ui()
        Gd()
    elif abs(cube[h][1] - cube[c][1]) == 2:
        U2()
        Gc()


def bottom_right_slot():
    """
    1*2 slot at bottom right
    possible cases: Rb, Ga, Gb
    """
    if abs(cube[c][1] - cube[h][1]) == 2:
        U()
        Rb()
    elif abs(cube[a][1] - cube[f][1]) == 2:
        Ga()
    elif abs(cube[c][0] - cube[a][0]) == 2:
        Ui()
        Gb()


def last():
    """
    one last step: align the solved last layer with the correct center pieces
    """
    if cube[e][1] == 18:
        U()
    elif cube[b][0] == 18:
        U2()
    elif cube[d][1] == 18:
        Ui()


def pll():
    """
    permute last layer (move all stickers to place)
    need only one algorithm
    """
    if cube == ref:
        # cube solved
        return True
    elif abs(cube[a][0] - cube[b][0]) == abs(cube[b][0] - cube[c][0]) == abs(cube[f][0] - cube[g][0]) == abs(cube[g][0] - cube[h][0]) == abs(cube[c][1] - cube[e][1]) == abs(cube[e][1] - cube[h][1]) == abs(cube[a][1] - cube[d][1]) == abs(cube[d][1] - cube[f][1]) == 1:
        # cube almost solved except the orientation of the entire upper layer, need to use last()
        last()
    else:
        # one entire slot completed - Ua, Ub, Ja, Jb, F
        # case1 - B slot completed
        if abs(cube[a][0] - cube[b][0]) == abs(cube[b][0] - cube[c][0]) == 1:
            back_slot_completed()
        # case2 - F slot completed
        elif abs(cube[f][0] - cube[g][0]) == abs(cube[g][0] - cube[h][0]) == 1:
            U2()
            back_slot_completed()
        # case3 - R slot completed
        elif abs(cube[c][1] - cube[e][1]) == abs(cube[e][1] - cube[h][1]) == 1:
            Ui()
            back_slot_completed()
        # case4 - L slot completed
        elif abs(cube[a][1] - cube[d][1]) == abs(cube[d][1] - cube[f][1]) == 1:
            U()
            back_slot_completed()
        # 2 by 2 square
        # case1 - bottom left
        elif abs(cube[f][0] - cube[g][0]) == 1 and abs(cube[f][1] - cube[d][1]) == 1:
            bottom_left_square()
        # case2 - top left
        elif abs(cube[a][0] - cube[b][0]) == 1 and abs(cube[a][1] - cube[d][1]) == 1:
            Ui()
            bottom_left_square()
        # case3 - top right
        elif abs(cube[c][0] - cube[b][0]) == 1 and abs(cube[c][1] - cube[e][1]) == 1:
            U2()
            bottom_left_square()
        # case4 - bottom right
        elif abs(cube[g][0] - cube[h][0]) == 1 and abs(cube[h][1] - cube[e][1]) == 1:
            U()
            bottom_left_square()
        # order of 4 edges are correct (except 'square' cases)
        # E difference between two opposite corner stickers (on opposite faces) = 2
        # H difference between two adjacent corner stickers (on same face) = 2
        elif cube[d][1] == 10 and cube[g][0] == 18 and cube[e][1] == 26 and cube[b][0] == 34:
            if abs(cube[a][0] - cube[f][0]) == 2:
                if abs(cube[d][1] - cube[f][1]) == 1:
                    E()
                else:
                    U()
                    E()
                    Ui()
            elif abs(cube[a][1] - cube[f][1]) == 2:
                U2()
                H()
        elif cube[d][1] == 18 and cube[g][0] == 26 and cube[e][1] == 34 and cube[b][0] == 10:
            if abs(cube[a][0] - cube[f][0]) == 2:
                if abs(cube[d][1] - cube[f][1]) == 1:
                    E()
                    Ui()
                else:
                    Ui()
                    E()
            elif abs(cube[a][1] - cube[f][1]) == 2:
                U()
                H()
        elif cube[d][1] == 26 and cube[g][0] == 34 and cube[e][1] == 10 and cube[b][0] == 18:
            if abs(cube[a][0] - cube[f][0]) == 2:
                if abs(cube[d][1] - cube[f][1]) == 1:
                    E()
                    U2()
                else:
                    U()
                    E()
                    U()
            elif abs(cube[a][1] - cube[f][1]) == 2:
                H()
        elif cube[d][1] == 34 and cube[g][0] == 10 and cube[e][1] == 18 and cube[b][0] == 26:
            if abs(cube[a][0] - cube[f][0]) == 2:
                if abs(cube[d][1] - cube[f][1]) == 1:
                    E()
                    U()
                else:
                    U()
                    E()
            elif abs(cube[a][1] - cube[f][1]) == 2:
                Ui()
                H()
        # Na and Nb
        elif abs(cube[g][0] - cube[h][0]) == abs(cube[a][0] - cube[b][0]) == abs(cube[c][1] - cube[e][1]) == abs(cube[d][1] - cube[f][1]) == 1:
            Na()
        elif abs(cube[g][0] - cube[f][0]) == abs(cube[c][0] - cube[b][0]) == abs(cube[h][1] - cube[e][1]) == abs(cube[d][1] - cube[a][1]) == 1:
            Nb()
        # 1*2 slot when at front layer, is on the left
        # case1 - bottom left
        elif abs(cube[f][0] - cube[g][0]) == 1:
            bottom_left_slot()
        # case2 - top left
        elif abs(cube[a][1] - cube[d][1]) == 1:
            Ui()
            bottom_left_slot()
        # case3 - top right
        elif abs(cube[b][0] - cube[c][0]) == 1:
            U2()
            bottom_left_slot()
        # case4 - bottom right
        elif abs(cube[e][1] - cube[h][1]) == 1:
            U()
            bottom_left_slot()
        # 1*2 slot when at front layer, is on the right
        # case1 - bottom right
        elif abs(cube[h][0] - cube[g][0]) == 1:
            bottom_right_slot()
        # case2 - bottom left
        elif abs(cube[d][1] - cube[f][1]) == 1:
            Ui()
            bottom_right_slot()
        # case3 - top left
        elif abs(cube[a][0] - cube[b][0]) == 1:
            U2()
            bottom_right_slot()
        # case4 - top right
        elif abs(cube[e][1] - cube[c][1]) == 1:
            U()
            bottom_right_slot()
        # last case - Z
        else:
            Z()
        last()


def simplify(lst):
    """
    to simplify movements in different ways
    """
    cont = True
    while cont:
        count = 0
        # cancel out adjacent counter movements X and X'
        for ni in range(len(lst)):
            if ni < len(lst) - 1:
                if len(lst[ni]) == 2:
                    if lst[ni + 1] == lst[ni][0]:
                        lst.pop(ni + 1)
                        lst.pop(ni)
                        count += 1
                    elif ni > 0 and lst[ni - 1] == lst[ni][0]:  # ni>0: cannot cancel out lst[0] and lst[-1]
                        lst.pop(ni)
                        lst.pop(ni - 1)
                        count += 1
        for ni in range(len(lst)):
            if ni < len(lst) - 2 and lst[ni] == lst[ni + 1] == lst[ni + 2]:
                # three identical moves XXX equals to its counter move X'
                if "'" in lst[ni]:
                    lst[ni] = lst[ni][:-1]
                else:
                    lst[ni] += "'"
                lst.pop(ni + 2)
                lst.pop(ni + 1)
                count += 1
            if ni < len(lst) - 1 and lst[ni] == lst[ni + 1]:
                # write double turns XX as X2
                lst[ni] = lst[ni][0] + "2"
                lst.pop(ni + 1)
                count += 1
        if count == 0:  # cannot be further simplified
            cont = False
        else:  # possible to be further simplified
            cont = True
    return lst


# valid scramble moves
valid_moves = ("U", "U'", "U2", "F", "F'", "F2", "R", "R'", "R2", "L", "L'", "L2", "B", "B'", "B2", "D", "D'", "D2")


def user_scramble():
    """
    Ask user to enter the scramble at any length
    :return: updated moves list with scramble in it
    """
    cont = True
    while cont:
        user = input("***** Please use the notation: U U' U2 F F' F2 R R' R2 L L' L2 B B' B2 D D' D2 (NO M E S)*****\nEnter your scramble (separate by comma, NO space) or 'done' to exit: ")
        if user.lower() == 'done':
            raw = []
            cont = False
        else:
            if user.count(" ") != 0:
                print("There should be NO space in between moves")
                cont = True
            else:
                raw = user.split(",")
                error = 0
                for ni in raw:
                    if ni not in valid_moves:
                        error += 1
                        print("Wrong notation")
                        break
                if error == 0:
                    cont = False
                else:
                    cont = True
    if raw == []:
        return False
    else:
        simplify(raw)
        for ni in raw:
            if ni == "U":
                U()
            elif ni == "U'":
                Ui()
            elif ni == "U2":
                U2()
            elif ni == "F":
                F()
            elif ni == "F'":
                Fi()
            elif ni == "F2":
                F2()
            elif ni == "R":
                R()
            elif ni == "R'":
                Ri()
            elif ni == "R2":
                R2()
            elif ni == "L":
                L()
            elif ni == "L'":
                Li()
            elif ni == "L2":
                L2()
            elif ni == "B":
                B()
            elif ni == "B'":
                Bi()
            elif ni == "B2":
                B2()
            elif ni == "D":
                D()
            elif ni == "D'":
                Di()
            elif ni == "D2":
                D2()
        simplify(moves)
    return True


def cube_solver():
    """solve a 3 by 3 cube using CFOP"""
    cont = True
    while cont:
        moves.clear()  # clear the moves list to start a new scramble and solution
        user_scramble()
        if cube == ref:
            print("Unscrambled Cube")
            cont = False
        else:
            scramble = moves.copy()  # only scramble moves (simplified)
            extra = len(scramble)
            cross()
            first_layer()
            second_layer()
            oll1()
            oll2()
            pll()
            solution = moves[extra:]  # take out scramble, leave only solution moves
            simplify(solution)
            print(" ".join(solution))
            if cube == ref:
                print('SOLVED!')
            while True:
                another = input("Would you like to solve another cube? Enter 'y' for yes or 'n' for no: ")
                if another.lower() == 'y':
                    break
                elif another.lower() == 'n':
                    cont = False
                    break
                else:
                    print('Invalid answer')
                    continue

                
cube_solver()


'''
Test Cases:

D,L,U,B,R',L2,U,F',R,B',L2,F',R',D,U',R,B2,U',D',R'

D',U2,F2,L,U,L2,U2,B,U,B,L2,D,F',U,L',D,B',D',U2,B'

B',L,D2,R',L2,F2,R,F',B,D2,U,B,U2,L2,U',B',D2,F',B,D

L,F',D2,F2,R',D',B,D2,F2,R,F2,R,D2,L',B,R,U,B2,R,D2

F',B,U',R2,U',D',L,F',R',U2,R',L2,F2,L2,B,F,U,F,B',D'

D2,U,R2,F',L,U,D2,B',F2,L2,D2,U,F',B,R,L2,B,R',U,F
'''
