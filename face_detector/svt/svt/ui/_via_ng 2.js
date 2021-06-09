/*
  VGG Image Annotator (VIA)
  www.robots.ox.ac.uk/~vgg/software/via/

  Copyright (c) 2016-2017, Abhishek Dutta, Visual Geometry Group, Oxford University.
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are met:

  Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.
  Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.
  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
  POSSIBILITY OF SUCH DAMAGE.
*/


////////////////////////////////////////////////////////////////////////////////
//
// @file        _via_region.js
// @description Implementation of region shapes like rectangle, circle, etc.
// @author      Abhishek Dutta <adutta@robots.ox.ac.uk>
// @date        17 June 2017
//
////////////////////////////////////////////////////////////////////////////////

function _via_region( shape, id, data_view_space, scale_factor ) {
  // Note the following terminology:
  //   view space  :
  //     - corresponds to the x-y plane on which the scaled version of original image is shown to the user
  //     - all the region query operations like is_inside(), is_on_edge(), etc are performed in view space
  //     - all svg draw operations like get_svg() are also in view space
  //
  //   image space :
  //     - corresponds to the x-y plane which corresponds to the spatial space of the original image
  //     - region save, export, git push operations are performed in image space
  //     - to avoid any rounding issues (caused by floating scale factor),
  //        * user drawn regions in view space is first converted to image space
  //        * this region in image space is now used to initialize region in view space
  //
  //   The two spaces are related by _via_model.now.tform.scale which is computed by the method
  //     _via_ctrl.compute_view_panel_to_nowfile_tform()
  //   and applied as follows:
  //     x coordinate in image space = scale_factor * x coordinate in view space
  //
  // shape : {rect, circle, ellipse, line, polyline, polygon, point}
  // id    : unique region-id
  // d[]   : (in view space) data whose meaning depend on region shape as follows:
  //        rect     : d[x1,y1,x2,y2] or d[corner1_x, corner1_y, corner2_x, corner2_y]
  //        circle   : d[x1,y1,x2,y2] or d[center_x, center_y, circumference_x, circumference_y]
  //        ellipse  : d[x1,y1,x2,y2]
  //        line     : d[x1,y1,x2,y2]
  //        polyline : d[x1,y1,...,xn,yn]
  //        polygon  : d[x1,y1,...,xn,yn]
  //        point    : d[cx,cy,r]
  // scale_factor : for conversion from view space to image space
  //
  // Note: no svg data are stored with prefix "_". For example: _scale_factor, _x2
  this.shape  = shape;
  this.id     = id;
  this.scale_factor     = scale_factor;
  this.scale_factor_inv = 1.0 / this.scale_factor;
  this.recompute_svg    = false;
  this.attributes  = {};

  var n = data_view_space.length;
  this.dview  = new Array(n);
  this.dimg   = new Array(n);

  if ( n !== 0 ) {
    // IMPORTANT:
    // to avoid any rounding issues (caused by floating scale factor), we stick to
    // the principal that image space coordinates are the ground truth for every region.
    // Hence, we proceed as:
    //   * user drawn regions in view space is first converted to image space
    //   * this region in image space is now used to initialize region in view space
    for ( var i = 0; i < n; i++ ) {
      this.dimg[i]  = Math.round( data_view_space[i] * this.scale_factor );
      this.dview[i] = Math.round( this.dimg[i] * this.scale_factor_inv );
    }
  }

  switch( this.shape ) {
  case "rect":
    _via_region_rect.call( this );
    this.svg_attributes = ['x', 'y', 'width', 'height'];
    break;
  case "circle":
    _via_region_circle.call( this );
    this.svg_attributes = ['cx', 'cy', 'r'];
    break;
  case "ellipse":
    _via_region_ellipse.call( this );
    this.svg_attributes = ['cx', 'cy', 'rx', 'ry'];
    break;
  case "line":
    _via_region_line.call( this );
    this.svg_attributes = ['x1', 'y1', 'x2', 'y2'];
    break;
  case "polyline":
    _via_region_polyline.call( this );
    this.svg_attributes = ['points'];
    break;
  case "polygon":
    _via_region_polygon.call( this );
    this.svg_attributes = ['points'];
    break;
  case "point":
    _via_region_point.call( this );
    // point is a special circle with minimal radius required for visualization
    this.shape = 'circle';
    this.svg_attributes = ['cx', 'cy', 'r'];
    break;
  }

  this.initialize();
}


_via_region.prototype.prepare_svg_element = function() {
  var _VIA_SVG_NS = "http://www.w3.org/2000/svg";
  this.svg_element = document.createElementNS(_VIA_SVG_NS, this.shape);
  this.svg_string  = '<' + this.shape;
  this.svg_element.setAttributeNS(null, 'id', this.id);

  var n = this.svg_attributes.length;
  for ( var i = 0; i < n; i++ ) {
    this.svg_element.setAttributeNS(null, this.svg_attributes[i], this[this.svg_attributes[i]]);
    this.svg_string += ' ' + this.svg_attributes[i] + '="' + this[this.svg_attributes[i]] + '"';
  }
  this.svg_string  += '/>';
}

_via_region.prototype.get_svg_element = function() {
  if ( this.recompute_svg ) {
    this.prepare_svg_element();
    this.recompute_svg = false;
  }
  return this.svg_element;
}

_via_region.prototype.get_svg_string = function() {
  if ( this.recompute_svg ) {
    this.prepare_svg_element();
    this.recompute_svg = false;
  }
  return this.svg_string;
}

_via_region.prototype.move_dview = function( dx, dy ) {
  // IMPORTANT:
  // to avoid any rounding issues (caused by floating scale factor), we stick to
  // the principal that image space coordinates are the ground truth for every region.
  // Hence, we proceed as:
  //   * user drawn regions in view space is first converted to image space
  //   * this region in image space is now used to initialize region in view space

  var n = this.dview.length;
  for ( var i = 0; i < n; i += 2 ) {
    var view_x = this.dview[i] + dx;
    var view_y = this.dview[i+1] + dy;
    this.dimg[i]   = Math.round( view_x * this.scale_factor );
    this.dimg[i+1] = Math.round( view_y * this.scale_factor );

    this.dview[i]   = Math.round( this.dimg[i] * this.scale_factor_inv );
    this.dview[i+1] = Math.round( this.dimg[i+1] * this.scale_factor_inv );
  }
}
_via_region.prototype.set_vertex_dview = function( vertex, x, y ) {
  // IMPORTANT:
  // to avoid any rounding issues (caused by floating scale factor), we stick to
  // the principal that image space coordinates are the ground truth for every region.
  // Hence, we proceed as:
  //   * user drawn regions in view space is first converted to image space
  //   * this region in image space is now used to initialize region in view space

  var i = 2 * vertex;
  this.dimg[i]   = Math.round( x * this.scale_factor );
  this.dimg[i+1] = Math.round( y * this.scale_factor );

  this.dview[i]   = Math.round( this.dimg[i]   * this.scale_factor_inv );
  this.dview[i+1] = Math.round( this.dimg[i+1] * this.scale_factor_inv );
}

_via_region.prototype.set_vertex_x_dview = function (vertex, x) {
  var i = 2 * vertex;
  this.dimg[i]   = Math.round( x * this.scale_factor );
  this.dview[i]  = Math.round( this.dimg[i]   * this.scale_factor_inv );
}

_via_region.prototype.set_vertex_y_dview = function (vertex, y) {
  var i = 2 * vertex;
  this.dimg[i+1]   = Math.round( y * this.scale_factor );
  this.dview[i+1]  = Math.round( this.dimg[i+1]   * this.scale_factor_inv );
}

///
/// Region shape : rectangle
///
function _via_region_rect() {
  this.is_inside  = _via_region_rect.prototype.is_inside;
  this.is_on_edge = _via_region_rect.prototype.is_on_edge;
  this.move  = _via_region_rect.prototype.move;
  this.resize  = _via_region_rect.prototype.resize;
  this.initialize = _via_region_rect.prototype.initialize;
  this.dist_to_nearest_edge = _via_region_rect.prototype.dist_to_nearest_edge;
}

_via_region_rect.prototype.initialize = function() {
  // ensure that this.(x,y) corresponds to top-left corner of rectangle
  // Note: this.(x2,y2) is defined for convenience in calculations
  if ( this.dview[0] < this.dview[2] ) {
    this.x  = this.dview[0];
    this.x2 = this.dview[2];
  } else {
    this.x  = this.dview[2];
    this.x2 = this.dview[0];
  }
  if ( this.dview[1] < this.dview[3] ) {
    this.y  = this.dview[1];
    this.y2 = this.dview[3];
  } else {
    this.y  = this.dview[3];
    this.y2 = this.dview[1];
  }
  this.width  = this.x2 - this.x;
  this.height = this.y2 - this.y;
  this.recompute_svg = true;
}

_via_region_rect.prototype.is_inside = function( px, py ) {
  if ( px > this.x && px < this.x2 ) {
    if ( py > this.y && py < this.y2 ) {
      return true;
    }
  }
  return false;
}

_via_region_rect.prototype.is_on_edge = function( px, py, tolerance ) {
  // -1 indicates that (px,py) is neither on edge nor on corners
  // 1 : corner - top left
  // 2 : edge   - top
  // 3 : corner - top right
  // 4 : edge   - right
  // 5 : corner - bottom right
  // 6 : edge   - bottom
  // 7 : corner - bottom left
  // 8 : edge   - left
  var dx0 = Math.abs( this.x  - px );
  var dy0 = Math.abs( this.y  - py );
  var dx2 = Math.abs( this.x2 - px );
  var dy2 = Math.abs( this.y2 - py );

  if ( dx0 < tolerance && dy2 < tolerance ) {
    return 7;
  }

  if ( dx2 < tolerance && dy0 < tolerance ) {
    return 3;
  }

  if ( dx0 < tolerance ) {
    if ( dy0 < tolerance ) {
      return 1;
    } else {
      if ( py > this.y && py < this.y2 ) {
        return 8;
      }
    }
  } else {
    if ( dy0 < tolerance ) {
      if ( px > this.x && px < this.x2 ) {
        return 2;
      }
    }
  }

  if ( dx2 < tolerance ) {
    if ( dy2 < tolerance ) {
      return 5;
    } else {
      if ( py < this.y2 && py > this.y ) {
        return 4;
      }
    }
  } else {
    if ( dy2 < tolerance ) {
      if ( px > this.x && px < this.x2 ) {
        return 6;
      }
    }
  }

  return -1; // not on edge
}

_via_region_rect.prototype.dist_to_nearest_edge = function( px, py ) {
  // given the coordinates of two opposite corners of a rectangle
  // (dview[0], dview[1]) and (dview[2], dview[3])
  // plot the following points and you will see why this works
  var d = [Math.abs(py - this.dview[1]),
           Math.abs(py - this.dview[3]),
           Math.abs(px - this.dview[0]),
           Math.abs(px - this.dview[2])];
  return Math.min.apply(null, d);
}

_via_region_rect.prototype.move = function(dx, dy) {
  this.move_dview(dx, dy);
  this.initialize();
}

_via_region_rect.prototype.resize = function(vertex, x, y) {
  // see _via_region_rect() for definition of constants
  switch (vertex) {
  case 1:
    this.set_vertex_dview(0, x, y);
    break;
  case 2:
    this.set_vertex_y_dview(0, y);
    break;
  case 3:
    this.set_vertex_y_dview(0, y);
    this.set_vertex_x_dview(1, x);
    break;
  case 4:
    this.set_vertex_x_dview(1, x);
    break;
  case 5:
    this.set_vertex_dview(1, x, y);
    break;
  case 6:
    this.set_vertex_y_dview(1, y);
    break;
  case 7:
    this.set_vertex_x_dview(0, x);
    this.set_vertex_y_dview(1, y);
    break;
  case 8:
    this.set_vertex_x_dview(0, x);
    break;
  }
  this.initialize();
}

///
/// Region shape : circle
///
function _via_region_circle() {
  this.is_inside  = _via_region_circle.prototype.is_inside;
  this.is_on_edge = _via_region_circle.prototype.is_on_edge;
  this.move       = _via_region_circle.prototype.move;
  this.resize     = _via_region_circle.prototype.resize;
  this.initialize = _via_region_circle.prototype.initialize;
  this.dist_to_nearest_edge = _via_region_circle.prototype.dist_to_nearest_edge;
}

_via_region_circle.prototype.initialize = function() {
  this.cx = this.dview[0];
  this.cy = this.dview[1];
  var dx = this.dview[2] - this.dview[0];
  var dy = this.dview[3] - this.dview[1];
  this.r  = Math.round( Math.sqrt(dx * dx + dy * dy) );
  this.r2 = this.r * this.r;
  this.recompute_svg = true;
}

_via_region_circle.prototype.is_inside = function( px, py ) {
  var dx = px - this.cx;
  var dy = py - this.cy;
  if ( ( dx * dx + dy * dy ) < this.r2 ) {
    return true;
  } else {
    return false;
  }
}

_via_region_circle.prototype.is_on_edge = function( px, py, tolerance ) {
  // -1 indicates that (px,py) is neither on edge nor on corners
  // 1 : corner - top left (-00 to -70 degrees)
  // 2 : edge   - top (-0 to -00 degrees)
  // 3 : corner - top right (-10 to -80 degrees)
  // 4 : edge   - right (-10 to +10 degrees)
  // 5 : corner - bottom right (10 to 80 degrees)
  // 6 : edge   - bottom (80 to 100 degrees)
  // 7 : corner - bottom left (100 to 170 degrees)
  // 8 : edge   - left (>170 and < -170 degrees)
  var tlist = [ 0.174, 1.396, 1.745, 2.967 ];
  var dx = px - this.cx;
  var dy = py - this.cy;
  var dxdy2 = dx*dx + dy*dy;

  var ra = this.r - tolerance;
  var rb = this.r + tolerance;

  if ( dxdy2 >= (ra*ra) && dxdy2 <= (rb*rb) ) {
    var t = Math.atan2( py - this.cy, px - this.cx );
    if ( t >=  tlist[0] && t <=  tlist[1] ) {
      return 5;
    }
    if ( t >= -tlist[1] && t <= -tlist[0] ) {
      return 3;
    }
    if ( t >=  tlist[2] && t <=  tlist[3] ) {
      return 7;
    }
    if ( t >= -tlist[3] && t <= -tlist[2] ) {
      return 1;
    }

    if ( t >= -tlist[0] && t <=  tlist[0] ) {
      return 4;
    }
    if ( t >=  tlist[1] && t <=  tlist[2] ) {
      return 6;
    }
    if ( t >=  tlist[3] || t <= -tlist[3] ) {
      return 8;
    }
    if ( t >= -tlist[2] && t <= -tlist[1] ) {
      return 2;
    }
  }
  return -1;
}

_via_region_circle.prototype.dist_to_nearest_edge = function( px, py ) {
  // distance to edge = radius - distance_to_point
  var dx = px - this.cx;
  var dy = py - this.cy;
  var dp = Math.sqrt(dx * dx + dy * dy);
  return this.r - dp;
}

_via_region_circle.prototype.move = function( dx, dy ) {
  this.move_dview( dx, dy );
  this.initialize();
}

_via_region_circle.prototype.resize = function( vertex, x_new, y_new ) {
  // whatever is the vertex, we always resize the circle boundary
  this.set_vertex_dview(1, x_new, y_new);
  this.initialize();
}

///
/// Region shape : ellipse
///
function _via_region_ellipse() {
  this.is_inside  = _via_region_ellipse.prototype.is_inside;
  this.is_on_edge = _via_region_ellipse.prototype.is_on_edge;
  this.move  = _via_region_ellipse.prototype.move;
  this.resize  = _via_region_ellipse.prototype.resize;
  this.initialize = _via_region_ellipse.prototype.initialize;
  this.dist_to_nearest_edge = _via_region_ellipse.prototype.dist_to_nearest_edge;
}

_via_region_ellipse.prototype.initialize = function() {
  this.cx = this.dview[0];
  this.cy = this.dview[1];
  this.rx = Math.abs(this.dview[2] - this.dview[0]);
  this.ry = Math.abs(this.dview[3] - this.dview[1]);

  this.inv_rx2 = 1 / (this.rx * this.rx);
  this.inv_ry2 = 1 / (this.ry * this.ry);

  this.recompute_svg = true;
}

_via_region_ellipse.prototype.is_inside = function(px, py) {
  var dx = this.cx - px;
  var dy = this.cy - py;

  if ( ( (dx * dx * this.inv_rx2) + (dy * dy * this.inv_ry2) ) < 1 ) {
    return true;
  } else {
    return false;
  }
}

_via_region_ellipse.prototype.is_on_edge = function( px, py, tolerance ) {
  // -1 indicates that (px,py) is neither on edge nor on corners
  // 1 : corner - top left (-00 to -70 degrees)
  // 2 : edge   - top (-0 to -00 degrees)
  // 3 : corner - top right (-10 to -80 degrees)
  // 4 : edge   - right (-10 to +10 degrees)
  // 5 : corner - bottom right (10 to 80 degrees)
  // 6 : edge   - bottom (80 to 100 degrees)
  // 7 : corner - bottom left (100 to 170 degrees)
  // 8 : edge   - left (>170 and < -170 degrees)
  var tlist = [ 0.174, 1.396, 1.745, 2.967 ];

  var d1rx = this.rx - tolerance;
  var d1ry = this.ry - tolerance;
  var d2rx = this.rx + tolerance;
  var d2ry = this.ry + tolerance;

  var inv_d1rx2 = 1 / (d1rx * d1rx);
  var inv_d1ry2 = 1 / (d1ry * d1ry);
  var inv_d2rx2 = 1 / (d2rx * d2rx);
  var inv_d2ry2 = 1 / (d2ry * d2ry);

  var dx = px - this.cx;
  var dy = py - this.cy;
  var dx2 = dx * dx;
  var dy2 = dy * dy;

  if ( (dx2 * inv_d1rx2) + (dy2 * inv_d1ry2) >= 1 &&
       (dx2 * inv_d2rx2) + (dy2 * inv_d2ry2) <= 1
     ) {
    var t = Math.atan2(dy, dx);
    if ( t >=  tlist[0] && t <=  tlist[1] ) {
      return 5;
    }
    if ( t >= -tlist[1] && t <= -tlist[0] ) {
      return 3;
    }
    if ( t >=  tlist[2] && t <=  tlist[3] ) {
      return 7;
    }
    if ( t >= -tlist[3] && t <= -tlist[2] ) {
      return 1;
    }

    if ( t >= -tlist[0] && t <=  tlist[0] ) {
      return 4;
    }
    if ( t >=  tlist[1] && t <=  tlist[2] ) {
      return 6;
    }
    if ( t >=  tlist[3] || t <= -tlist[3] ) {
      return 8;
    }
    if ( t >= -tlist[2] && t <= -tlist[1] ) {
      return 2;
    }
  }
  return -1;
}

_via_region_ellipse.prototype.dist_to_nearest_edge = function( px, py ) {
  // stretch the line connecting ellipse center and (px,py) towards the ellipse edge
  // length of this line is given by the polar form of ellipse relative to center
  // for more details: see https://en.wikipedia.org/wiki/Ellipse#Polar_forms
  var dx = px - this.cx;
  var dy = py - this.cy;
  var h = Math.sqrt( dx*dx + dy*dy ); // hypotenuse
  var hinv = 1 / h;
  var p = py - this.cy; // perpendicular
  var b = px - this.cx; // base
  var rx_sint = this.rx * p * hinv;
  var ry_cost = this.ry * b * hinv;
  var sqterm = Math.sqrt( ry_cost*ry_cost + rx_sint*rx_sint );
  var r_theta = (this.rx * this.ry ) / sqterm;
  return Math.abs(r_theta - h);
}

_via_region_ellipse.prototype.move = function(dx, dy) {
  this.move_dview(dx, dy);
  this.initialize();
}

_via_region_ellipse.prototype.resize = function(vertex, x, y) {
  switch (vertex) {
  case 2:
  case 6:
    this.set_vertex_y_dview(1, y);
    break;
  case 4:
  case 8:
    this.set_vertex_x_dview(1, x);
    break;
  }
  this.initialize();
}

///
/// Region shape : line
///
function _via_region_line() {
  this.is_inside  = _via_region_line.prototype.is_inside;
  this.is_on_edge = _via_region_line.prototype.is_on_edge;
  this.move  = _via_region_line.prototype.move;
  this.resize  = _via_region_line.prototype.resize;
  this.initialize = _via_region_line.prototype.initialize;
  this.dist_to_nearest_edge = _via_region_line.prototype.dist_to_nearest_edge;
}

_via_region_line.prototype.initialize = function() {
  this.x1 = this.dview[0];
  this.y1 = this.dview[1];
  this.x2 = this.dview[2];
  this.y2 = this.dview[3];
  this.dx = this.x1 - this.x2;
  this.dy = this.y1 - this.y2;
  this.mconst = (this.x1 * this.y2) - (this.x2 * this.y1);

  this.recompute_svg = true;
}

_via_region_line.prototype.is_inside = function( px, py ) {
  // compute the area of a triangle made up of the following three vertices
  // (x1,y1) (x2,y2) and (px,py)
  //        | px py 1 |
  // Area = | x1 y1 1 | = px(y1 - y2) - py(x1 - x2) + (x1*y2 - x2*y1)
  //        | x2 y2 1 |
  var area = Math.abs( px*this.dy - py*this.dx + this.mconst );
  var area_tolerance = Math.abs( 5 * (this.dy + this.dx) ); // area diff. when (x1,y1) moved by 5 pixel

  if ( area <= area_tolerance ) {
    // check if (px,py) lies between (x0,y0) and (x1,y1)
    if ( (px > this.x1 && px < this.x2) || (px < this.x1 && px > this.x2) ) {
      if ( (py > this.y1 && py < this.y2) || (py < this.y1 && py > this.y2) ) {
        return true;
      }
    }
  } else {
    return false;
  }
}

_via_region_line.prototype.dist_to_nearest_edge = function( px, py ) {
  var dx = this.x2 - this.x1;
  var dy = this.y2 - this.y1;
  // see https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line#Line_defined_by_two_points
  var x2y1 = this.x2 * this.y1;
  var y2x1 = this.y2 * this.x1;
  var denominator = Math.sqrt( dx*dx + dy*dy );
  var numerator   = Math.abs( dy*px - dx*py + x2y1 - y2x1 );
  return ( numerator / denominator );
}

_via_region_line.prototype.is_on_edge = function( px, py, tolerance ) {
  // -1 indicates that (px,py) is not on edge
  // N  indicates near vertex N
  var tolerance2 = tolerance * tolerance;
  var dx1 = px - this.x1;
  var dy1 = py - this.y1;
  var dx2 = px - this.x2;
  var dy2 = py - this.y2;

  if (dx1*dx1 + dy1*dy1 <= tolerance2) {
    return 0;
  }

  if (dx2*dx2 + dy2*dy2 <= tolerance2) {
    return 1;
  }

  return -1;
}

_via_region_line.prototype.move = function( dx, dy ) {
  this.move_dview(dx, dy);
  this.initialize();
}

_via_region_line.prototype.resize = function(vertex, x, y) {
  this.set_vertex_dview(vertex, x, y);
  this.initialize();
}

///
/// Region shape : polyline
///
function _via_region_polyline() {
  this.is_inside  = _via_region_polyline.prototype.is_inside;
  this.is_on_edge = _via_region_polyline.prototype.is_on_edge;
  this.move  = _via_region_polyline.prototype.move;
  this.resize  = _via_region_polyline.prototype.resize;
  this.initialize = _via_region_polyline.prototype.initialize;
  this.dist_to_nearest_edge = _via_region_polyline.prototype.dist_to_nearest_edge;
}

_via_region_polyline.prototype.initialize = function() {
  var n = this.dview.length;
  var points = new Array(n/2);
  var points_index = 0;
  for ( var i = 0; i < n; i += 2 ) {
    points[points_index] = ( this.dview[i] + ' ' + this.dview[i+1] );
    points_index++;
  }
  this.points = points.join(',');
  this.recompute_svg = true;
}

_via_region_polyline.prototype.is_inside = function( px, py ) {
  // @todo: optimize
  var tolerance = 10;
  var n = this.dview.length;

  // check if (px,py) is near any vertex
  for ( var i = 0; i < n; i += 2 ) {
    var dx = this.dview[i] - px;
    var dy = this.dview[i+1] - py;

    if ( (dx*dx + dy*dy) <= tolerance ) {
      return true;
    }
  }

  var n = this.dview.length - 2;
  for ( var i = 0; i < n; i += 2 ) {
    var x0 = this.dview[i];
    var y0 = this.dview[i+1];
    var x1 = this.dview[i+2];
    var y1 = this.dview[i+3];

    var dx = x0 - x1;
    var dy = y0 - y1;
    var mconst = x0*y1 - x1*y0;
    var area = Math.abs( px*dy - py*dx + mconst );
    var area_tolerance = Math.abs( 5*(dx+dy) );
    if ( area <= area_tolerance ) {
      // check if (px,py) lies between (x0,y0) and (x1,y1)
      if ( (px > x0 && px < x1) || (px < x0 && px > x1) ) {
        if ( (py > y0 && py < y1) || (py < y0 && py > y1) ) {
          return true;
        }
      }
    }
  }
  return false;
}

_via_region_polyline.prototype.dist_to_nearest_edge = function( px, py ) {
  var n = this.dview.length;
  var vertex = 0;
  var distances = [];
  for ( var i = 0; i < n; i += 2 ) {
    var dx = this.dview[i+2] - this.dview[i];
    var dy = this.dview[i+3] - this.dview[i+1];
    // see https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line#Line_defined_by_two_points
    var x2y1 = this.dview[i+2] * this.dview[1];
    var y2x1 = this.dview[i+3] * this.dview[0];
    var denominator = Math.sqrt( dx*dx + dy*dy );
    var numerator   = Math.abs( dy*px - dx*py + x1y1 - y2x1 );
    distances.push( numerator / denominator );
  }
  return Math.min.apply(null, distances);
}

_via_region_polyline.prototype.is_on_edge = function( px, py, tolerance ) {
  // -1 indicates that (px,py) is not on edge
  // return vertex index when (px,py) is within tolerance of a vertex (not edge)
  var n = this.dview.length;
  var vertex = 0;
  for ( var i = 0; i < n; i += 2 ) {
    var dx = this.dview[i] - px;
    var dy = this.dview[i+1] - py;

    if ( (dx*dx + dy*dy) <= tolerance ) {
      return vertex;
    }
    vertex++;
  }
  return -1;
}

_via_region_polyline.prototype.move = function( dx, dy ) {
  this.move_dview(dx, dy);
  this.initialize();
}

_via_region_polyline.prototype.resize = function( vertex, x, y ) {
  this.set_vertex_dview(vertex, x, y);
  this.initialize();
}

///
/// Region shape : polygon
///
function _via_region_polygon() {
  this.is_inside  = _via_region_polygon.prototype.is_inside;
  this.is_on_edge = _via_region_polygon.prototype.is_on_edge;
  this.move  = _via_region_polygon.prototype.move;
  this.resize  = _via_region_polygon.prototype.resize;
  this.initialize = _via_region_polygon.prototype.initialize;
  this.dist_to_nearest_edge = _via_region_polygon.prototype.dist_to_nearest_edge;
}

_via_region_polygon.prototype.initialize = function() {
  var n = this.dview.length;
  var points = new Array(n/2);
  var points_index = 0;
  for ( var i = 0; i < n; i += 2 ) {
    points[points_index] = ( this.dview[i] + ' ' + this.dview[i+1] );
    points_index++;
  }
  this.points = points.join(',');
  this.recompute_svg = true;
}

_via_region_polygon.prototype.is_inside = function( px, py ) {
  // ref: http://geomalgorithms.com/a03-_inclusion.html
  var n = this.dview.length;

  var wn = 0;    // the  winding number counter
  // loop through all edges of the polygon
  for ( var i = 0; i < n; i += 2 ) {   // edge from V[i] to  V[i+1]
    var x0 = this.dview[i];
    var y0 = this.dview[i+1];
    var x1 = this.dview[i+2];
    var y1 = this.dview[i+3];

    // area of triangle is 0 if points are collinear
    var is_left_value =  ((x1 - x0) * (py - y0)) - ((y1 - y0) * (px - x0));

    if ( y0 <= py ) {
      if ( y1  > py && is_left_value > 0) {
        ++wn;
      }
    }
    else {
      if ( y1  <= py && is_left_value < 0) {
        --wn;
      }
    }
  }
  if ( wn === 0 ) {
    return false;
  }
  else {
    return true;
  }
}

_via_region_polygon.prototype.dist_to_nearest_edge = function( px, py ) {
  // create a copy of vertices and add first vertex to close path
  var n = this.dview.length;
  var points = this.dview.slice(0);
  points.push(points[0]);
  points.push(points[1]);

  var vertex = 0;
  var distances = [];

  for ( var i = 0; i < n; i += 2 ) {
    var dx = points[i+2] - points[i];
    var dy = points[i+3] - points[i+1];

    // see https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line#Line_defined_by_two_points
    var x2y1 = points[i+2] * points[i+1];
    var y2x1 = points[i+3] * points[i];
    var denominator = Math.sqrt( dx*dx + dy*dy );
    var numerator   = Math.abs( dy*px - dx*py + x2y1 - y2x1 );

    distances.push( numerator / denominator );
  }
  return Math.min.apply(null, distances);
}

_via_region_polygon.prototype.is_on_edge = function( px, py, tolerance ) {
  // -1 indicates that (px,py) is not on edge
  // -1 indicates that (px,py) is not on edge
  // return vertex index when (px,py) is within tolerance of a vertex (not edge)
  var n = this.dview.length;
  var vertex = 0;
  for ( var i = 0; i < n; i += 2 ) {
    var dx = this.dview[i] - px;
    var dy = this.dview[i+1] - py;

    if ( (dx*dx + dy*dy) <= tolerance ) {
      return vertex;
    }
    vertex++;
  }
  return -1;
}

_via_region_polygon.prototype.move = function( dx, dy ) {
  this.move_dview(dx, dy);
  this.initialize();
}

_via_region_polygon.prototype.resize = function( vertex, x, y ) {
  this.set_vertex_dview(vertex, x, y);
  this.initialize();
}

///
/// Region shape : point
///
function _via_region_point() {
  this.is_inside  = _via_region_point.prototype.is_inside;
  this.is_on_edge = _via_region_point.prototype.is_on_edge;
  this.move  = _via_region_point.prototype.move;
  this.resize  = _via_region_point.prototype.resize
  this.initialize  = _via_region_point.prototype.initialize;
  this.dist_to_nearest_edge = _via_region_point.prototype.dist_to_nearest_edge;
}

_via_region_point.prototype.initialize = function() {
  this.cx = this.dview[0];
  this.cy = this.dview[1];
  this.r  = 3;
  this.r2 = this.r * this.r;
  this.recompute_svg = true;
}

_via_region_point.prototype.is_inside = function( px, py ) {
  var dx = px - this.cx;
  var dy = py - this.cy;
  if ( ( dx * dx + dy * dy ) < this.r2 ) {
    return true;
  } else {
    return false;
  }
}

_via_region_point.prototype.dist_to_nearest_edge = function( px, py ) {
  var dx = px - this.cx;
  var dy = py - this.cy;
  return Math.sqrt( dx * dx + dy * dy );
}

_via_region_point.prototype.is_on_edge = function( px, py, tolerance ) {
  // -1 indicates that (px,py) is not on edge
  return -1;
}

_via_region_point.prototype.move = function( dx, dy ) {
  this.move_dview(dx, dy);
  this.initialize();
}

_via_region_point.prototype.resize = function( vertex, dx, dy ) {
  return;
}

////////////////////////////////////////////////////////////////////////////////
//
// @file        _via_model.js
// @description Implements Model in the MVC architecture of VIA
// @author      Abhishek Dutta <adutta@robots.ox.ac.uk>
// @date        17 June 2017
//
////////////////////////////////////////////////////////////////////////////////

function _via_model() {
  this.about = {};
  this.about.name = 'VGG Image Annotator';
  this.about.shortname = 'VIA';
  this.about.version = '2.0.0';

  this.regions = {};

  this.repo = {};
  this.repo.local = {};
  this.repo.remote = {};

  this.files = {};
  this.files.metadata = {};
  this.files.content = {};
  this.files.fileid_list = [];
  this.files.file_index_list = {};
}

_via_model.prototype.init = function( via_ctrl ) {
  //console.log('Initializing _via_model ...');
  this.c = via_ctrl;
}

///
/// add/remove files being annotated
///
_via_model.prototype.add_file_from_base64 = function(filename, data_base64) {
  var filename   = filename;
  var filesize   = data_base64.length;
  var frame      = 0;
  var count      = 1;
  var filesource = 'base64';
  var filetype   = data_base64.substr(5, 5);
  var filecontent= data_base64;

  return new Promise( function(ok_callback, err_callback) {
    this.add_file(filename, filesize, frame, count, filesource, filetype, filecontent).then( function(fileid) {
      ok_callback(fileid);
    }, function(error_msg) {
      err_callback(error_msg);
    });
  }.bind(this));
}

_via_model.prototype.add_file_from_url = function(url, type) {
  var filename   = url;
  var filesize   = -1;
  var frame      = 0;
  var count      = 1;
  var filesource = 'url';
  var filetype   = type;
  var filecontent= url;

  return new Promise( function(ok_callback, err_callback) {
    this.add_file(filename, filesize, frame, count, filesource, filetype, filecontent).then( function(fileid) {
      ok_callback(fileid);
    }.bind(this), function(error_msg) {
      err_callback(error_msg);
    }.bind(this));
  }.bind(this));
}

_via_model.prototype.add_file = function(filename, filesize, frame, count, filesource, filetype, filecontent) {
  // filename    : local filename or image url
  // filesize    : file size in bytes (-1 when filesource is url)
  // frame       : frame index in video (0 for image)
  // count       : number of frames starting from from 'frame' (1 for image)
  // filetype    : {image, video} (0 for unknown)
  // filesource  : {local, url, base64}
  // filecontent : local file reference, url, base64 data
  //
  return new Promise( function(ok_callback, err_callback) {
    this.c.file_hash(filename, filesize, frame, count).then( function(fileid) {
      if ( this.files.fileid_list.includes(fileid) ) {
        this.c.show_message('File ' + filename + ' already loaded. Skipping!');
	err_callback('Error : file already loaded : ' + filename);
      }
      this.files.file_index_list[fileid] = this.files.fileid_list.length;
      this.files.fileid_list.push(fileid);

      this.files.content[fileid] = filecontent;

      this.files.metadata[fileid] = {};
      this.files.metadata[fileid].fileid   = fileid;
      this.files.metadata[fileid].filename = filename;
      this.files.metadata[fileid].filesize = filesize;
      this.files.metadata[fileid].frame    = frame;
      this.files.metadata[fileid].count    = count;
      this.files.metadata[fileid].source   = filesource
      this.files.metadata[fileid].type     = filetype;

      this.regions[fileid] = {};
      ok_callback(fileid);
    }.bind(this), function(error) {
      err_callback('Error : error computing unique hash for file : '
                   + filename + '{error: ' + error + '}');
    }.bind(this));
  }.bind(this));
}

_via_model.prototype.add_file_local = function(file) {
  var filename   = file.name;
  var filesize   = file.size;
  var frame      = 0;
  var count      = 1;
  var filesource = 'local';
  var filetype   = file.type.substr(0, 5);
  var filecontent= file

  return new Promise( function(ok_callback, err_callback) {
    this.add_file(filename, filesize, frame, count, filesource, filetype, filecontent).then( function(fileid) {
      ok_callback(fileid);
    }, function(error_msg) {
      err_callback(error_msg);
    });
  }.bind(this));
}

///
/// Region management (add/update/delete/...)
///
_via_model.prototype.region_add = function(fileid, tform_scale, shape, nvertex) {
  // unique region id is constructed as follows:
  //   shape + Date.now() + first_vertex
  // where,
  // shape : {rect, circle, ellipse, line, polyline, polygon, point}
  // Date.now() : number of milliseconds elapsed since epoch
  // first_vertex : 3 digit coordinate padded with 0
  /*
  return new Promise( function( ok_callback, err_callback ) {
    //var id   = this.m.settings.useremail + '_' + Date.now() + '_' + name;
    var rid   = shape + '_' + Date.now() + '_' + ('000' + nvertex[0]).slice(-3);
    var region = new _via_region(shape, rid, nvertex, tform_scale);
    this.regions[fileid][rid] = region;

    ok_callback(rid);
  }.bind(this) );
  */
  var rid   = shape + '_' + Date.now() + '_' + ('000' + nvertex[0]).slice(-3);
  var region = new _via_region(shape, rid, nvertex, tform_scale);
  this.regions[fileid][rid] = region;
  return rid;
}

_via_model.prototype.region_del = function(fileid, rid) {
  return new Promise( function( ok_callback, err_callback ) {
    //var id   = this.m.settings.useremail + '_' + Date.now() + '_' + name;
    if ( this.regions[fileid].hasOwnProperty(rid) ) {
      delete this.regions[fileid][rid];
      ok_callback(rid);
    } else {
      err_callback(rid);
    }
  }.bind(this) );
}

_via_model.prototype.region_move = function(fileid, rid, dx, dy) {
  return new Promise( function( ok_callback, err_callback ) {
    this.regions[fileid][rid].move( dx, dy );
    ok_callback( {fileid: fileid, region_id: rid} );
  }.bind(this) );
}

_via_model.prototype.region_resize = function(fileid, rid, vertex, x_new, y_new) {
  return new Promise( function( ok_callback, err_callback ) {
    this.regions[fileid][rid].resize( vertex, x_new, y_new );
    ok_callback( {fileid: fileid, region_id: rid} );
  }.bind(this) );
}

///
/// Region query
///
_via_model.prototype.is_point_in_a_region = function(fileid, px, py) {
  // returns region_id if (px,py) is inside a region
  // otherwise returns ''

  var rid_list = [];
  for( var rid in this.regions[fileid] ) {
    if ( this.regions[fileid].hasOwnProperty(rid) ) {
      if ( this.regions[fileid][rid].is_inside( px, py ) ) {
        rid_list.push( rid );
      }
    }
  }

  switch( rid_list.length ) {
  case 0:
    return '';
    break;
  case 1:
    return rid_list[0];
    break;
  default:
    return this.region_nearest_to_point(px, py, rid_list, fileid);
    break;
  }

  // @todo : for large number of polygons, employ more efficient algorithms
  // using the Slab Decomposition algorithm
  // ref: https://en.wikipedia.org/wiki/Point_location#Slab_decomposition
}

_via_model.prototype.region_nearest_to_point = function(px, py, rid_list, fileid) {
  var min_edge_distance = Number.MAX_VALUE;
  var min_edge_index = -1;
  for ( var i=0; i<rid_list.length; i++ ) {
    var rid = rid_list[i];
    var edge_distance = this.regions[fileid][rid].dist_to_nearest_edge(px,py);
    if ( edge_distance < min_edge_distance ) {
      min_edge_distance = edge_distance;
      min_edge_index = i;
    }
  }
  return rid_list[ min_edge_index ];
}

_via_model.prototype.is_point_inside_region = function(fileid, rid, px, py) {
  return this.regions[fileid][rid].is_inside( px, py );
}

// returns edge-id if (px,py) is on a selected region edge
// otherwise -1
_via_model.prototype.is_on_region_edge = function(fileid, rid, px, py, tolerance) {
  return this.regions[fileid][rid].is_on_edge( px, py, tolerance );
}

_via_model.prototype.is_on_these_region_edge = function(fileid, rid_list, px, py, tolerance) {
  var n = rid_list.length;
  for ( var i=0; i<n; i++ ) {
    var rid  = rid_list[i];
    var edge = this.regions[fileid][rid].is_on_edge( px, py, tolerance );
    if ( edge !== -1 ) {
      return {'rid':rid, 'id':edge};
    }
  }
  return {'rid':-1, 'id':-1};
}

_via_model.prototype.is_point_inside_these_regions = function(fileid, rid_list, px, py) {
  var n = rid_list.length;
  var candidate_rid_list = [];

  for ( var i=0; i<n; i++ ) {
    var rid = rid_list[i];
    if ( this.regions[fileid][rid].is_inside( px, py ) ) {
      candidate_rid_list.push( rid );
    }
  }

  switch( candidate_rid_list.length ) {
  case 0:
    return '';
    break;
  case 1:
    return candidate_rid_list[0];
    break;
  default:
    return this.region_nearest_to_point(px, py, candidate_rid_list, fileid);
    break;
  }
}

///
/// metadata
///

////////////////////////////////////////////////////////////////////////////////
//
// @file        _via_view.js
// @description Implements View in the MVC architecture of VIA
// @author      Abhishek Dutta <adutta@robots.ox.ac.uk>
// @date        17 June 2017
//
////////////////////////////////////////////////////////////////////////////////


function _via_view() {

  this.state = {UNKNOWN: 'UNKNOWN',
                IDLE: 'IDLE',
                REGION_SELECTED: 'REGION_SELECTED',
                REGION_SELECT_OR_DRAW_POSSIBLE: 'REGION_SELECT_OR_DRAW_POSSIBLE',
                SELECT_ALL_INSIDE_AN_AREA_ONGOING: 'SELECT_ALL_INSIDE_AN_AREA_ONGOING',
                REGION_UNSELECT_ONGOING: 'REGION_UNSELECT_ONGOING',
                REGION_SELECT_TOGGLE_ONGOING: 'REGION_SELECT_TOGGLE_ONGOING',
                REGION_MOVE_ONGOING: 'REGION_MOVE_ONGOING',
                REGION_RESIZE_ONGOING: 'REGION_RESIZE_ONGOING',
                REGION_DRAW_ONGOING: 'REGION_DRAW_ONGOING',
                REGION_DRAW_NCLICK_ONGOING: 'REGION_DRAW_NCLICK_ONGOING',
                FILE_LOAD_ONGOING: 'FILE_LOAD_ONGOING'
               };
  this.state_now = this.state.UNKNOWN;

  this.layers = {}; // content layers in this.v.view_panel

  // record of the immediate past events
  this.last = {};
  this.last.mousemove = {};
  this.last.mousedown = {};
  this.last.mouseup   = {};
  this.nvertex = []; // array of _via_point

  this.now = {};
  // these fileds are populated during runtime:
  // this.now.fileid
  // this.now.tform.{x,y,width,height,scale,content_width, content_height}
  // this.now.{all_rid_list,polygon_rid_list,other_rid_list}
  // this.now.meta_keypress.{shiftkey, ctrlkey, altkey}
  // this.now.region_select.rid_list

  // User views a scaled version of original image (that fits in the web
  // browser display panel). The tform parameters are needed to convert
  // the user drawn regions in the web browser to coordinates in the
  // original image dimension
  // For example: original_img_x = drawn_x * tform.scale;
  this.now.tform = {};

  this.now.region_shape = '';
  this.now.region_select = {};
  this.now.region_select.rid_list = [];
  this.now.region_select.fileid = '';

  this.now.view_tmp_region = {};
  this.now.view_tmp_region.rid_list = [];
  this.now.view_tmp_region.DEFAULT_RID = '_via_tmp_svg_region';
  this.now.view_tmp_region.RID_PREFIX_MOVE = '_via_tmp_move_';
  this.now.view_tmp_region.RID_PREFIX_RESIZE = '_via_tmp_resize_';


  this.now.all_rid_list = [];
  this.now.polygon_rid_list = [];
  this.now.other_rid_list = [];

  this.now.meta_keypress = {};
  this.now.meta_keypress.shiftkey = false;
  this.now.meta_keypress.altkey = false;
  this.now.meta_keypress.ctrlkey = false;

  this.now.content_metadata = {};
  this.now.content_metadata.type = '';
  this.now.content_metadata.is_paused = false;

  this.settings = {};
  this.settings.username = '_via_user_name';
  this.settings.useremail= '_via_user_email';
  this.settings.REGION_SHAPE =  {RECT    :'rect',
                                 CIRCLE  :'circle',
                                 ELLIPSE :'ellipse',
                                 POINT   :'point',
			         LINE    :'line',
                                 POLYGON :'polygon',
			         POLYLINE:'polyline'};

  this.settings.theme    = {};
  this.settings.theme.svg= {};
  this.settings.theme.svg.REGION = { 'fill': 'none',
                                     'fill-opacity': '0',
                                     'stroke': 'yellow',
                                     'stroke-width': '2'
                                   };
  this.settings.theme.svg.SELECTED_REGION = { 'fill': 'white',
                                              'fill-opacity': '0.2',
                                              'stroke': 'black',
                                              'stroke-width': '2'
                                            };
  this.settings.theme.svg.ON_DRAWING = { 'fill': 'none',
                                         'fill-opacity': '0',
                                         'stroke': 'red',
                                         'stroke-width': '2'
                                       };
  this.settings.theme.svg.ON_MOVE = { 'fill': 'white',
                                      'fill-opacity': '0.2',
                                      'stroke': 'red',
                                      'stroke-width': '2'
                                    };

  this.settings.theme.svg.ON_SELECT_AREA_DRAW = { 'fill': 'none',
                                                  'stroke': 'red',
                                                  'stroke-width': '2',
                                                  'stroke-dasharray': '5,5'
                                                };
  // point is a circle with radius 1, the radius is set higher for visualization
  this.settings.theme.svg.POINT_SHAPE_DRAW_RADIUS = 2;

  this.settings.theme.MSG_TIMEOUT_MS = 5000;
  this.settings.theme.MOUSE_CLICK_TOLERANCE2    = 4; // 2 pixels
  this.settings.theme.REGION_ON_EDGE_TOLERANCE2 = 4; // 2 pixels
  this.settings.theme.THETA_TOLERANCE           = Math.PI / 18; // 10 degrees
  this.settings.theme.MOUSE_VERTEX_TOLERANCE = 10; // pixels
  this.settings.theme.FILE_VIEW_SCALE_MODE = 'fit'; // {original, fit}

  // zoom
  this.zoom = {};
  this.zoom.container = {};
  this.zoom.container_id = 'zoom_panel';
  this.zoom.filecontent = {};

  this.zoom.is_enabled = false;
  this.zoom.size = 300; // a square region with size in pixels
  this.zoom.sizeby2 = this.zoom.size/2;
  this.zoom.scale = 2.0; // 1.0 for no scale
  this.zoom.DEFAULT_SCALE = 2.0;

  //this.local_file_selector : created by init_local_file_selector()

  // notify _via_ctrl if user interaction events
  this.handleEvent = function(e) {
    switch(e.currentTarget.id) {
    case 'rect':
    case 'circle':
    case 'ellipse':
    case 'polygon':
    case 'line':
    case 'polyline':
    case 'point':
      this.c.set_region_shape(e.currentTarget.id);
      break;

    case 'add_file_local':
    case 'menubar_add_file_local':
      this.c.select_local_files();
      break;

    case 'move_to_prev_file':
      this.c.load_prev_file();
      break;
    case 'move_to_next_file':
      this.c.load_next_file();
      break;

    case 'zoom_in':
      this.c.zoom_activate();
      break;
    case 'zoom_reset':
      this.c.zoom_deactivate();
      break;

    case 'region_delete_selected':
      this.c.delete_selected_regions();
      break;

    case 'region_select_all':
      this.c.region_select_all();
      break;

    default:
      console.log('_via_view: handler unknown for event: ' + e.currentTarget);
    }
    e.stopPropagation();
    //this.c.layers['top'].focus();
  }
}


///
/// state maintainance
///
_via_view.prototype.set_state = function(state) {
  if ( this.state.hasOwnProperty( state ) ) {
    this.state_now = state;
    //console.log('State = ' + this.state_now);
  } else {
    console.log('_via_view.prototype.set_state() :: Cannot set to unknown state: ' + state);
  }
}

_via_view.prototype.init = function( via_ctrl, view_panel, message_panel ) {
  //console.log('Initializing _via_view ...');
  this.c = via_ctrl;

  if( typeof(view_panel) === 'undefined' ) {
    console.log('Error: _via_view.prototype.init() : view panel has not been defined');
  } else {
    this.view_panel = view_panel;
  }

  if( typeof(message_panel) === 'undefined' ) {
    _via_ctrl.prototype.show_message = function(msg, t){};
  } else {
    this.message_panel = message_panel;
  }

  this.view_panel.innerHTML = '';

  this.init_local_file_selector();
  this.register_ui_action_handlers();

  //this.init_metadata_io_panel_drag_handler();
}

_via_view.prototype.init_local_file_selector = function() {
  this.local_file_selector = document.createElement('input');
  this.local_file_selector.setAttribute('id', 'local_file_selector');
  this.local_file_selector.setAttribute('type', 'file');
  this.local_file_selector.setAttribute('name', 'files[]');
  this.local_file_selector.setAttribute('multiple', 'multiple');
  this.local_file_selector.style.display = 'none';
  this.local_file_selector.setAttribute('accept', '.jpg,.jpeg,.png,.bmp,.mp4,.ogg,.webm');
  this.local_file_selector.addEventListener('change',
                                            this.c.add_user_sel_local_files.bind(this.c),
                                            false);

  this.view_panel.appendChild(this.local_file_selector);
}

_via_view.prototype.register_ui_action_handlers = function() {
  // defined in javascript embedded in file _via_view.html
}

////////////////////////////////////////////////////////////////////////////////
//
// @file        _via_ctrl.js
// @description Implements Controller in the MVC architecture of VIA
// @author      Abhishek Dutta <adutta@robots.ox.ac.uk>
// @date        17 June 2017
//
////////////////////////////////////////////////////////////////////////////////

function _via_ctrl() {
  this.hook = {};
  this.hook.id = {REGION_ADDED: 'REGION_ADDED',
                  REGION_DRAWN: 'REGION_DRAWN',
                  REGION_MOVED: 'REGION_MOVED',
                  REGION_RESIZED: 'REGION_RESIZED',
                  FILE_LOAD_FINISH: 'FILE_LOAD_FINISH',
                  CLICKED_AT: 'CLICKED_AT',
                  VIDEO_ON_PLAY: 'VIDEO_ON_PLAY',
                  VIDEO_ON_PAUSE: 'VIDEO_ON_PAUSE',
                  VIDEO_ON_SEEK: 'VIDEO_ON_SEEK'
                  };

  this.hook.now = {};
}

function _via_point(x, y) {
  this.x = x;
  this.y = y;
}

///
/// initialization routines
///
_via_ctrl.prototype.init = function(via_model, via_view) {
  //console.log('Initializing _via_ctrl ...');
  this.m = via_model;
  this.v = via_view;

  this.init_defaults();
  this.init_view_panel();

  // add user input handlers
  this.v.layers.top.addEventListener('keydown', this.keydown_handler.bind(this), false);
  this.v.layers.top.addEventListener('keyup', this.keyup_handler.bind(this), false);

  this.v.set_state( this.v.state.IDLE );
  this.v.layers.top.focus();

  this.init_message_panel();
}

_via_ctrl.prototype.init_defaults = function() {
  this.set_region_shape('rect');
}

///
/// hook maintainers
///
_via_ctrl.prototype.add_hook = function(hook_name, listener) {
  if ( this.hook.now.hasOwnProperty(hook_name) ) {
    this.hook.now[hook_name].push(listener);
  } else {
    this.hook.now[hook_name] = [];
    this.hook.now[hook_name].push(listener);
  }
}

_via_ctrl.prototype.del_hook = function(hook_name, listener) {
  if ( this.hook.now[hook_name].includes(listener) ) {
    var index = this.hook.now[hook_name].indexOf(listener);
    this.hook.now[hook_name].splice(index, 1);
  }
}

_via_ctrl.prototype.clear_all_hooks = function(hook_name) {
  this.hook.now[hook_name] = [];
}

_via_ctrl.prototype.trigger_hook = function(hook_name, param) {
  return new Promise( function(ok_callback, err_callback) {
    if ( this.hook.now.hasOwnProperty(hook_name) ) {
      for ( var i=0; i<this.hook.now[hook_name].length; i++ ) {
        var target = this.hook.now[hook_name][i];
        target.call(null, param);
      }
    }
  }.bind(this));
}

///
/// maintainers of the view panel
///
_via_ctrl.prototype.init_view_panel = function() {
  // ensure that all the layers fill up the parent container
  this.v.view_panel.style.width = '100%';
  this.v.view_panel.style.height = '100%';
  this.v.view_panel.width  = this.v.view_panel.offsetWidth;
  this.v.view_panel.height = this.v.view_panel.offsetHeight;

  this.v.layers = {};
  this.v.layers['filecontent'] = document.createElement('div');
  this.v.layers['filecontent'].setAttribute('id', 'filecontent');
  //this.v.layers['filecontent'].style.zIndex = "9971";

  this.v.layers['rshape_canvas'] = document.createElement('canvas');
  this.v.layers['rshape_canvas'].setAttributeNS(null, 'id', 'rshape_canvas');

  this.v.layers['rshape'] = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  this.v.layers['rshape'].setAttributeNS('http://www.w3.org/2000/xmlns/', 'xmlns:xlink', 'http://www.w3.org/2000/svg');
  this.v.layers['rshape'].setAttributeNS(null, 'id', 'rshape');
  //this.v.layers['rshape'].style.zIndex = "9972";

  this.v.layers['rattr'] = document.createElement('div');
  this.v.layers['rattr'].setAttribute('id', 'rattr');
  //this.v.layers['rattr'].style.zIndex = "9973";

  this.v.layers['zoom'] = document.createElement('div');
  this.v.layers['zoom'].setAttribute('id', 'zoom');
  //this.v.layers['zoom'].style.zIndex = "9974";

  this.v.layers['top'] = document.createElement('div');
  this.v.layers['top'].setAttribute('id', 'top');
  //this.v.layers['top'].setAttribute('tabindex', '1');
  //this.v.layers['top'].style.zIndex = "9980";

  for ( var layer in this.v.layers ) {
    if ( this.v.layers.hasOwnProperty(layer) ) {
      this.v.view_panel.appendChild(this.v.layers[layer]);
    }
  }

  /*
  // debug
  this.v.layers['filecontent'].addEventListener('mousemove', function(e) {
    console.log('filecontent: mousemove');
  });
  this.v.layers['rshape'].addEventListener('mousemove', function(e) {
    console.log('rshape: mousemove');
  });
  this.v.layers['zoom'].addEventListener('mousemove', function(e) {
    console.log('zoom: mousemove');
  });
  this.v.layers['rattr'].addEventListener('mousemove', function(e) {
    console.log('rattr: mousemove');
  });
  this.v.layers['top'].addEventListener('mousemove', function(e) {
    console.log('top: mousemove');
  });
  */

  // attach mouse and touch event handlers
  this.v.layers['top'].addEventListener('mousemove', this.mousemove_handler.bind(this));
  this.v.layers['top'].addEventListener('mouseup'  , this.mouseup_handler.bind(this));
  this.v.layers['top'].addEventListener('mousedown', this.mousedown_handler.bind(this));
  this.v.layers['top'].addEventListener('mouseover', this.mouseover_handler.bind(this));
  this.v.layers['top'].addEventListener('mouseout',  this.mouseout_handler.bind(this));

  this.v.layers['top'].addEventListener('touchstart', this.touchstart_handler);
  this.v.layers['top'].addEventListener('touchend'  , this.touchend_handler);
  this.v.layers['top'].addEventListener('touchmove' , this.touchmove_handler);
}


///
/// File load routines
///
_via_ctrl.prototype.load_next_file = function() {
  var filecount = this.m.files.fileid_list.length;
  if ( filecount > 0) {
    var now_file_index = this.m.files.file_index_list[this.v.now.fileid];
    if ( now_file_index === (filecount-1) ) {
      this.load_file_from_index(0);
    } else {
      this.load_file_from_index(now_file_index + 1);
    }
  }
}

_via_ctrl.prototype.load_prev_file = function() {
  var filecount = this.m.files.fileid_list.length;
  if ( filecount > 0) {
    var now_file_index = this.m.files.file_index_list[this.v.now.fileid];
    if ( now_file_index === 0 ) {
      this.load_file_from_index( filecount - 1);
    } else {
      this.load_file_from_index(now_file_index - 1);
    }
  }
}

_via_ctrl.prototype.load_file_from_index = function( file_index ) {
  if ( file_index >= 0 || file_index < this.m.files.fileid_list.length ) {
    var fileid = this.m.files.fileid_list[file_index];
    this.load_file(fileid);
  }
}

_via_ctrl.prototype.load_file = function( fileid ) {
  if ( this.v.state_now === this.v.state.FILE_LOAD_ONGOING ) {
    this.show_message('Please wait until the current file load operation completes!');
    return;
  }

  if ( this.v.state_now !== this.v.state.IDLE ) {
    this.show_message('Cannot load new file when state is ' + this.v.state_now +
                      '. Press [Esc] to reset state');
    return;
  }

  if ( this.m.files.fileid_list.includes( fileid ) ) {
    this.v.set_state( this.v.state.FILE_LOAD_ONGOING );

    // 1. Load file contents (local image, url, base64)
    this.set_now_file(fileid).then( function() {
      this.compute_view_panel_to_nowfile_tform();
      this.update_file_info_for_nowfile();

      this.v.set_state( this.v.state.IDLE );
      this.update_layers_size_for_nowfile();

      // maintain a list of regions for current file
      this.v.now.all_rid_list = [];
      this.v.now.polygin_rid_list = [];
      this.v.now.other_rid_list = [];
      for ( var rid in this.m.regions[fileid] ) {
        if ( this.m.regions[fileid].hasOwnProperty(rid) ) {
          // maintain a list of regions for current file
          this.v.now.all_rid_list.push(rid);
          switch(this.m.regions[fileid][rid].shape) {
          case this.v.settings.REGION_SHAPE.POLYGON:
          case this.v.settings.REGION_SHAPE.POLYLINE:
            // region shape requiring more than two points (polygon, polyline)
            this.v.now.polygon_rid_list.push(rid);
            break;
          default:
            // region shapes requiring just two points (rectangle, circle, etc)
            this.v.now.other_rid_list.push(rid);
            break;
          }
        }
      }
      // load layer content, region shape and region attributes
      this.load_layer_content();
      this.load_layer_rshape();
      this.load_layer_rattr();

      this.trigger_hook(this.hook.id.FILE_LOAD_FINISH, {'fileid': fileid});
    }.bind(this), function(error) {
      this.v.set_state( this.v.state.IDLE );
      var filename = decodeURIComponent(this.m.files.metadata[fileid].filename);
      this.show_message('Error loading file : ' + filename);
      console.log(error);
    }.bind(this));
  }
}


///
/// Register handlers for events generated by the VIEW
///
_via_ctrl.prototype.set_region_shape = function(shape) {
  // deactivate the current shape button
  if ( this.v.now.region_shape ) {
    var old_html_element = document.getElementById( this.v.now.region_shape );
    if ( old_html_element ) {
      if ( old_html_element.classList.contains('active') ) {
        old_html_element.classList.remove('active');
      }
    }
  }

  this.v.now.region_shape = shape;
  var new_html_element = document.getElementById( this.v.now.region_shape );
  if ( new_html_element ) {
    if ( !new_html_element.classList.contains('active') ) {
      new_html_element.classList.add('active');
    }
  }

  // activate the new shape button
  //console.log('Setting region shape to : ' + this.m.now.region_shape);
}

///
/// file add/remove
///
_via_ctrl.prototype.add_user_sel_local_files = function(event) {
  event.stopPropagation();

  var user_selected_files = event.target.files;
  var all_promise = [];

  for ( var i = 0; i < user_selected_files.length; ++i ) {
    var promise = this.m.add_file_local( user_selected_files[i] );
    all_promise.push( promise );
  }

  var loaded_images_count    = 0;
  var loaded_videos_count    = 0;
  var discarded_files_count  = 0;

  Promise.all( all_promise ).then( function(result) {
    var first_fileid = '';
    for ( var i=0; i<result.length; i++ ) {
      if ( result[i].startsWith('Error' ) ) {
        discarded_files_count++;
      } else {
        var fileid = result[i];
        switch( this.m.files.metadata[fileid].type ) {
        case 'image':
	  loaded_images_count++;
	  break;
        case 'video':
	  loaded_videos_count++;
	  break;
        }

        if ( first_fileid === '' ) {
          first_fileid = fileid;
        }
      }
    }
    var msg = 'Loaded ' + loaded_images_count + ' images' +
	' and ' + loaded_videos_count + ' videos';
    if ( discarded_files_count !== 0 ) {
      msg += ' ( ' + discarded_files_count + ' discarded )';
    }
    this.show_message(msg);

    // the user is automatically taken to the first loaded file
    if ( first_fileid !== '' ) {
      this.load_file( first_fileid );
    }
  }.bind(this), function(error_msg) {
    console.log(error_msg);
  }.bind(this));

}

///
/// prompts for user input in web browser
///
_via_ctrl.prototype.select_local_files = function() {
  // ref: https://developer.mozilla.org/en-US/docs/Using_files_from_web_applications
  this.v.local_file_selector.click();
}

///
/// routines to update and maintain _via_view_panel layers
///
_via_ctrl.prototype.compute_view_panel_to_nowfile_tform = function() {
  // content's original width and height
  var cw, ch;
  switch( this.m.files.metadata[this.v.now.fileid].type ) {
  case 'image':
    cw = this.v.now.content.width;
    ch = this.v.now.content.height;
    break;
  case 'video':
    cw = this.v.now.content.videoWidth;
    ch = this.v.now.content.videoHeight;
    break;
  default:
    cw = 0;
    ch = 0;
    this.show_message('Cannot detemine the width and height of file content : ' +
                      this.m.files.metadata[this.m.now.fileid].filename);
  }

  if ( this.v.settings.theme.FILE_VIEW_SCALE_MODE === 'original' ) {
    this.v.now.tform.x = 0;
    this.v.now.tform.y = 0;
    this.v.now.tform.width  = cw;
    this.v.now.tform.height = ch;
    this.v.now.tform.content_width = cw;
    this.v.now.tform.content_height = ch;
    this.v.now.tform.scale = 1.0;
    this.v.now.tform.scale_inv = 1.0;
  } else {
    // content layer width and height
    var lw = this.v.view_panel.offsetWidth;
    var lh = this.v.view_panel.offsetHeight;

    // transformed content's position, width and height
    var txw = cw;
    var txh = ch;
    var txx = 0;
    var txy = 0;

    var scale_width = lw / cw;
    txw = lw;
    txh = ch * scale_width;

    if ( txh > lh ) {
      var scale_height = lh / txh;
      txh = lh;
      txw = txw * scale_height;
    }

    /*
    // scale the image only if the image size is greater than layer
    if ( cw > lw ) {
    // reduce content size so that it fits layer width
    var scale_width = lw / cw;
    txw = lw;
    txh = ch * scale_width;
    }

    if ( txh > lh ) {
    // content still does not fit the layer height
    // scale the height further
    var scale_height = lh / txh;
    txh = lh;
    txw = txw * scale_height;
    }
    */

    /*  */
    // determine the position of content on content_layer
    this.v.now.tform.x = Math.floor( (lw - txw)/2 ); // align to center
    //this.v.now.tform.x = 0;
    this.v.now.tform.y = Math.floor( (lh - txh)/2 );
    this.v.now.tform.width  = Math.floor(txw);
    this.v.now.tform.height = Math.floor(txh);
    this.v.now.tform.content_width = cw;
    this.v.now.tform.content_height = ch;
    this.v.now.tform.scale = this.v.now.tform.content_width / this.v.now.tform.width;
    this.v.now.tform.scale_inv = 1.0 / this.v.now.tform.scale;
  }
}

_via_ctrl.prototype.update_layers_size_for_nowfile = function() {
  var style = [];
  style.push('display:inline-block');
  style.push('position:absolute');
  style.push('overflow:hidden'); // needed for zoom-panel overflowing in boundary

  //style.push('top:  ' + this.v.now.tform.y + 'px'); // vertical alignment = center
  style.push('top:0px'); // vertical alignment = top
  //style.push('left:' + this.v.now.tform.x + 'px');
  style.push('left:0px');
  style.push('width:' + this.v.now.tform.width  + 'px');
  style.push('height:' + this.v.now.tform.height + 'px');
  //style.push('outline:none');

  var style_str = style.join(';');

  for ( var layer in this.v.layers ) {
    if ( this.v.layers.hasOwnProperty(layer) ) {
      if ( layer !== 'filecontent' && this.m.files.metadata[this.v.now.fileid].type === 'video' ) {
        this.v.layers[layer].setAttribute('style', style_str + ';pointer-events:none;');
      } else {
        this.v.layers[layer].setAttribute('style', style_str);
      }
    }
  }

  this.v.layers['rshape_canvas'].width = this.v.now.tform.width;
  this.v.layers['rshape_canvas'].height = this.v.now.tform.height;
}

_via_ctrl.prototype.load_layer_content = function() {
  //this.v.layers['filecontent'].innerHTML = '';
  this.v.now.content.setAttribute('width', this.v.now.tform.width);
  this.v.now.content.setAttribute('height', this.v.now.tform.height);

  var child = this.v.layers['filecontent'].firstChild;
  if ( child ) {
    this.v.layers['filecontent'].replaceChild(this.v.now.content, child);
  } else {
    this.v.layers['filecontent'].appendChild(this.v.now.content);
  }
  /*
    if ( existing_content ) {
    this.v.layers['filecontent'].replaceChild(this.m.now.content, existing_content);
    } else {
    this.v.layers['filecontent'].appendChild(this.m.now.content);
    }
  */
}

_via_ctrl.prototype.load_layer_rshape = function() {
  // remove all existing regions
  this.remove_all_regions_from_view();

  // add regions from now
  var n = this.v.now.all_rid_list.length;
  for ( var i = 0; i < n; i++ ) {
    var rid = this.v.now.all_rid_list[i];
    this.add_region_to_view(rid);
  }
}

_via_ctrl.prototype.load_layer_rattr = function() {
  //@todo
}

///
/// routines to update and maintain general user interface elements
///
_via_ctrl.prototype.update_file_info_for_nowfile = function() {
  var now_fileindex = document.getElementById('now_fileindex');
  var now_filename  = document.getElementById('now_filename');

  if ( now_fileindex && now_filename ) {
    var file_index = this.m.files.file_index_list[this.v.now.fileid];
    var filename   = decodeURIComponent(this.m.files.metadata[this.v.now.fileid].filename);
    var file_count = this.m.files.fileid_list.length;
    //document.title = '[' + (file_index+1) + ' of ' + file_count + '] ' + this.m.files.metadata[fileid].filename;

    now_fileindex.innerHTML = (file_index + 1) + ' / ' + file_count;
    now_filename.innerHTML  = filename;
    now_filename.setAttribute('title', filename);
  }
}

_via_ctrl.prototype.init_message_panel = function() {
  this.v.message_panel.addEventListener('mousedown', function() {
    this.style.display = 'none';
  }, false);
  this.v.message_panel.addEventListener('mouseover', function() {
    clearTimeout(this.message_clear_timer); // stop any previous timeouts
  }, false);

}

_via_ctrl.prototype.show_message = function(msg, t) {
  if ( this.message_clear_timer ) {
    clearTimeout(this.message_clear_timer); // stop any previous timeouts
  }

  var timeout = t || this.v.settings.theme.MSG_TIMEOUT_MS
  this.v.message_panel.innerHTML = msg;
  this.v.message_panel.style.display = 'inline';
  if ( timeout !== 0 ) {
    this.message_clear_timer = setTimeout( function() {
      this.v.message_panel.innerHTML = '';
      this.v.message_panel.style.display = 'none';
    }.bind(this), timeout);
  }
}

///
/// keyboard input handlers
///
_via_ctrl.prototype.keyup_handler = function(e) {
  if (e.which === 16) {
    this.v.now.meta_keypress.shiftkey = false;
  }
  if ( e.which === 17 ) {
    this.v.now.meta_keypress.ctrlkey = false;
  }
  if (e.which === 18) {
    this.v.now.meta_keypress.altkey = false;
  }
}

_via_ctrl.prototype.keydown_handler = function(e) {
  e.stopPropagation();

  this.v.layers['top'].focus();
  e.preventDefault();

  if (e.which === 16) {
    this.v.now.meta_keypress.shiftkey = true;
  }
  if ( e.which === 17 ) {
    this.v.now.meta_keypress.ctrlkey = true;
  }
  if (e.which === 18) {
    this.v.now.meta_keypress.altkey = true;
  }

  // control commands
  if ( e.ctrlKey ) {
    this.v.now.meta_keypress.ctrlkey = true;
    if ( e.which === 83 ) { // Ctrl + s
      _via_ctrl_project_save_local();
      return;
    }
    if ( e.which === 79 ) { // Ctrl + o
      _via_ctrl_project_select_local_file();
      return;
    }
    if ( e.which === 65 ) { // Ctrl + a
      this.region_select_n( this.v.now.all_rid_list );
      this.v.set_state( this.v.state.REGION_SELECTED );
      return;
    }
    if ( e.which === 67 ) { // Ctrl + c
    }
    if ( e.which === 86 ) { // Ctrl + v
    }
  }

  if ( this.v.state_now == this.v.state.REGION_SELECTED ) {
    if( e.which === 46 || e.which === 8) { // Del or Backspace
      this.delete_selected_regions();
      return;
    }

    if ( e.which >= 37 && e.which <= 40 ) { // Arrow Keys
      // move all selected regions by 1 pixel
      var dx = 0;
      var dy = 0;
      // you can only move regions by 1 pixel (or, now.tform.scale) in the original image space
      if ( e.which === 39 ) { // right arrow
        dx = this.v.now.tform.scale_inv;
      }
      if ( e.which === 37 ) { // left arrow
        dx = -this.v.now.tform.scale_inv;
      }
      if ( e.which === 38 ) { // up arrow
        dy = -this.v.now.tform.scale_inv;
      }
      if ( e.which === 40 ) { // down arrow
        dy = this.v.now.tform.scale_inv;
      }

      if ( e.shiftKey ) {
        dx = dx * 10;
        dy = dy * 10;
      }

      if ( dx !== 0 || dy !== 0 ) {
        var n = this.v.now.region_select.rid_list.length;
        for ( var i=0; i<n; i++ ) {
          var fid = this.v.now.fileid;
          var rid = this.v.now.region_select.rid_list[i];

          this.m.region_move(fid, rid, dx, dy).then( function(result) {
	          this.remove_tmp_region_from_view();

	          if ( result.fileid === this.v.now.fileid ) {
              this.update_region_in_view( result.region_id );
	          }
          }.bind(this), function(error) {
            console.log(error);
          }.bind(this));
        }
        return;
      }
    }

    if ( e.which >= 27 ) { // Esc
      this.region_unselect_all();
      return;
    }
  }

  if ( this.v.state_now === this.v.state.IDLE ) {
    if ( e.which === 9 ) { // tab key
      if (e.shiftKey) {
        this.region_select_prev();
      } else {
        this.region_select_next();
      }
      return;
    }
  }

  if (e.which === 78 || e.which === 39 ) { // n or right arrow
    this.load_next_file();
    e.stopPropagation();
    return;
  }

  if (e.which === 80 || e.which === 37) { // p or left arrow
    this.load_prev_file();
    e.stopPropagation();
    return;
  }

  if (e.which === 121) { // F10 for debug messages
    e.stopPropagation();
    console.log('State = ' + this.v.state_now);
    console.log(this.m.regions[this.v.now.fileid]);
    console.log(this.m.files);
    console.log(this.v.now);
    return;
  }

  if (e.which === 13 ) { // Enter key
    e.stopPropagation();
    if ( this.v.state_now === this.v.state.REGION_DRAW_NCLICK_ONGOING ) {
      var fileid  = this.v.now.fileid;
      var rid = this.add_region_from_nvertex();
      this.trigger_hook(this.hook.id.REGION_DRAWN, {'fileid': fileid, 'rid': rid});
      this.v.nvertex.splice(0);
      this.v.set_state( this.v.state.IDLE );

    }
    return;
  }

  if (e.which === 27 ) { // Esc key
    e.stopPropagation();
    this.v.nvertex = [];
    this.remove_tmp_region_from_view();
    this.v.set_state( this.v.state.IDLE );
    return;
  }
}

///
/// Mouse and touch event hanlders
///

_via_ctrl.prototype.mousemove_handler = function(e) {
  e.stopPropagation();

  var x1 = e.offsetX;
  var y1 = e.offsetY;
  this.v.last.mousemove.x = x1;
  this.v.last.mousemove.y = y1;

  var nvertex_tmp = this.v.nvertex.slice(0);
  nvertex_tmp.push( x1 );
  nvertex_tmp.push( y1 );

  if(this.v.zoom.is_enabled) {
    this.update_zoom_panel_contents();
  }

  if ( this.v.state_now === this.v.state.REGION_DRAW_ONGOING ) {
    // region shape requiring than two points (rectangle, circle, ellipse, etc)
    this.add_tmp_region_to_view(nvertex_tmp);
    return;
  }

  if ( this.v.state_now === this.v.state.REGION_DRAW_NCLICK_ONGOING ) {
    // region shape that may require more than two points (polyline, polygon)
    this.add_tmp_region_to_view(nvertex_tmp);
    return;
  }

  if ( this.v.state_now === this.v.state.REGION_DRAW_OR_UNSELECT_POSSIBLE ) {
    var p0 = this.v.last.mousedown;
    var p1 = new _via_point(x1, y1);
    if ( !this.are_mouse_events_nearby( p0, p1 ) ) {
      this.region_unselect_all();
      this.v.set_state( this.v.state.REGION_DRAW_ONGOING );
    }
    return;
  }

  if ( this.v.state_now === this.v.state.REGION_MOVE_ONGOING ) {
    // region shape that may require more than two points (polyline, polygon)
    var dx = nvertex_tmp[2] - nvertex_tmp[0];
    var dy = nvertex_tmp[3] - nvertex_tmp[1];
    this.tmp_move_selected_region_in_view(dx, dy);
    return;
  }

  if ( this.v.state_now === this.v.state.REGION_RESIZE_ONGOING ) {
    // region shape that may require more than two points (polyline, polygon)
    var edge_id = this.v.last.clicked_region_edge_id;
    this.tmp_resize_selected_region_in_view(edge_id, x1, y1);
    return;
  }

  // mousedown inside an existing region and mouse being moved indicates region draw operation
  if ( this.v.state_now === this.v.state.REGION_SELECT_OR_DRAW_POSSIBLE ) {
    var p0 = this.v.last.mousedown;
    var p1 = new _via_point(x1, y1);
    if ( !this.are_mouse_events_nearby( p0, p1 ) ) {
      this.v.nvertex.push( this.v.last.mousedown.x );
      this.v.nvertex.push( this.v.last.mousedown.y );
      this.v.set_state( this.v.state.REGION_DRAW_ONGOING );
    }
    return;
  }

  if ( this.v.state_now === this.v.state.REGION_SELECTED ) {
    var fileid = this.v.now.fileid;
    var n = this.v.now.region_select.rid_list.length;
    for ( var i=0; i<n; i++ ) {
      var rid = this.v.now.region_select.rid_list[i];
      var tolerance = this.v.settings.theme.MOUSE_VERTEX_TOLERANCE;
      var edge_id = this.m.is_on_region_edge(fileid, rid, x1, y1, tolerance );
      if ( edge_id === -1 ) {
        // not on edge, is mouse inside the selected region?
        var is_inside_selected_region = this.m.is_point_inside_region(fileid, rid, x1, y1);
        if ( is_inside_selected_region ) {
          this.v.layers.top.style.cursor = 'move';
          return;
        }
      } else {
        // mouse cursor is on the edge
        var r = this.m.regions[this.v.now.fileid][rid];
        switch( r.shape ) {
        case this.v.settings.REGION_SHAPE.RECT:
        case this.v.settings.REGION_SHAPE.CIRCLE:
        case this.v.settings.REGION_SHAPE.ELLIPSE:
          switch( edge_id ) {
          case 1: // corner top-left
          case 5: // corner bottom-right
            this.v.layers.top.style.cursor = 'nwse-resize';
            break;
          case 3:
          case 7:
            this.v.layers.top.style.cursor = 'nesw-resize';
            break;
          case 2:
          case 6:
            this.v.layers.top.style.cursor = 'row-resize';
            break;
          case 4:
          case 8:
            this.v.layers.top.style.cursor = 'col-resize';
            break;
          }
          break;
        case this.v.settings.REGION_SHAPE.LINE:
        case this.v.settings.REGION_SHAPE.POLYLINE:
        case this.v.settings.REGION_SHAPE.POLYGON:
          //this.v.layers.top.style.cursor = 'move';
          this.v.layers.top.style.cursor = 'cell';
          break;
        }
        return;
      }
    }
    this.v.layers.top.style.cursor = 'default';
    return;
  }

  if ( this.v.state_now === this.v.state.SELECT_ALL_INSIDE_AN_AREA_ONGOING ) {
    this.add_tmp_region_to_view(nvertex_tmp,
                                this.v.settings.REGION_SHAPE.RECT,
                                this.v.settings.theme.svg.ON_SELECT_AREA_DRAW);
    return;
  }
}

_via_ctrl.prototype.mouseup_handler = function(e) {
  e.stopPropagation();

  var x1 = e.offsetX;
  var y1 = e.offsetY;
  this.v.last.mouseup = new _via_point(x1, y1);
  this.v.layers.top.focus();

  if ( this.v.state_now === this.v.state.REGION_DRAW_ONGOING ) {
    switch ( this.v.now.region_shape ) {
    case this.v.settings.REGION_SHAPE.POLYGON:
    case this.v.settings.REGION_SHAPE.POLYLINE:
      // region shape requiring more than two points (polygon, polyline)
      this.v.set_state( this.v.state.REGION_DRAW_NCLICK_ONGOING );
      break;

    default:
      // region shape requiring just two points (rectangle, circle, ellipse, etc)
      // check if the two vertices are malformed
      var p0 = this.v.last.mousedown;
      var p1 = this.v.last.mouseup;
      var is_click = this.are_mouse_events_nearby( p0, p1 );

      /*
      // hide the VIA layer to let event handler of video player handler mouse events
      if ( this.v.now.content_metadata.type === 'video' ) {
        if ( this.v.now.content_metadata.is_paused ) {
          this.v.layers['top'].style.pointerEvents = 'none';
        }
      }
      */

      if ( ! is_click || this.v.now.region_shape === this.v.settings.REGION_SHAPE.POINT ) {
        this.v.nvertex.push( x1 );
        this.v.nvertex.push( y1 );
        var fileid  = this.v.now.fileid;
        var rid = this.add_region_from_nvertex();
        this.trigger_hook(this.hook.id.REGION_DRAWN, {'fileid': fileid, 'rid': rid});
        this.remove_tmp_region_from_view();
      } else {
        var p0 = this.v.last.mousedown;
        // if no region is selected and user clicks outside any region,
        // it implies a mouse click
        if ( this.v.now.region_select.rid_list.length === 0) {
          this.trigger_hook(this.hook.id.CLICKED_AT, {'fileid': this.v.now.fileid, 'x':p0.x, 'y':p0.y});
        }
      }
      this.v.nvertex.splice(0);
      this.v.set_state( this.v.state.IDLE );
    }
    return;
  }

  if ( this.v.state_now === this.v.state.REGION_SELECT_OR_DRAW_POSSIBLE ) {
    // region select
    var rid = this.v.last.clicked_region_id;
    this.region_unselect_all();
    this.region_select(rid);
    this.v.now.region_select.fileid = this.v.now.fileid;
    this.v.set_state( this.v.state.REGION_SELECTED );

    if(this.v.zoom.is_enabled) {
      this.update_zoom_panel_contents();
    }

    this.show_message( 'Click and drag inside region to <span class="highlight">move</span>, ' +
                       'at region edge to <span class="highlight">resize</span>, ' +
		       'Move region by small amount using <span class="highlight">keyboard arrow keys</span>.');
  }

  if ( this.v.state_now === this.v.state.REGION_UNSELECT_ONGOING ) {
    this.region_unselect_all();
    this.v.set_state( this.v.state.IDLE );
    return;
  }

  if ( this.v.state_now === this.v.state.REGION_MOVE_ONGOING ) {
    this.v.nvertex.push( x1 );
    this.v.nvertex.push( y1 );
    var nvertex = this.v.nvertex.slice(0);

    var dx = nvertex[2] - nvertex[0];
    var dy = nvertex[3] - nvertex[1];

    var n = this.v.now.region_select.rid_list.length;
    for ( var i=0; i<n; i++ ) {
      var srid = this.v.now.region_select.rid_list[i];
      this.m.region_move(this.v.now.fileid, srid, dx, dy).then( function(result) {
        var srid_tmp = this.v.now.view_tmp_region.RID_PREFIX_MOVE + result.region_id;
        this.remove_tmp_region_from_view(srid_tmp);

        if ( result.fileid === this.v.now.fileid ) {
          this.update_region_in_view( result.region_id );
        }
        this.trigger_hook(this.hook.id.REGION_MOVED, {'fileid': result.fileid, 'rid': result.region_id});
      }.bind(this), function(error) {
        console.log(error);
      }.bind(this));
    }
    this.v.nvertex.splice(0);
    this.v.set_state( this.v.state.REGION_SELECTED );
    return;
  }

  if ( this.v.state_now === this.v.state.REGION_RESIZE_ONGOING ) {
    var edge_id = this.v.last.clicked_region_edge_id;
    var edge_rid = this.v.last.clicked_region_id;
    this.m.region_resize(this.v.now.fileid, edge_rid, edge_id, x1, y1).then( function(result) {
      var rid_tmp = this.v.now.view_tmp_region.RID_PREFIX_RESIZE + edge_rid;
      this.remove_tmp_region_from_view(rid_tmp);

      if ( result.fileid === this.v.now.fileid ) {
        this.update_region_in_view( result.region_id );
      }
      this.trigger_hook(this.hook.id.REGION_RESIZED, {'fileid': result.fileid, 'rid': result.region_id});
    }.bind(this), function(error) {
      console.log(error);
    }.bind(this));
    this.v.set_state( this.v.state.REGION_SELECTED );
    return;
  }

  if ( this.v.state_now === this.v.state.SELECT_ALL_INSIDE_AN_AREA_ONGOING ) {
    var x0 = this.v.last.mousedown.x;
    var y0 = this.v.last.mousedown.y;
    var rid_list = this.get_regions_inside_an_area(x0, y0, x1, y1);

    if ( rid_list.length ) {
      this.v.now.region_select.fileid = this.v.now.fileid;
      this.region_select_n( rid_list );
      this.v.set_state( this.v.state.REGION_SELECTED );
    } else {
      this.v.set_state( this.v.state.IDLE );
    }
    this.v.nvertex.splice(0);
    this.remove_tmp_region_from_view();
    return;
  }

  if ( this.v.state_now === this.v.state.REGION_SELECT_TOGGLE_ONGOING ) {
    var rid = this.v.last.clicked_region_id;
    this.region_select_toggle(rid);
    if ( this.v.now.region_select.rid_list.length === 0 ) {
      // region toggle resulted is no selected region
      this.v.set_state( this.v.state.IDLE );
      this.v.layers.top.style.cursor = 'default';
    } else {
      this.v.set_state( this.v.state.REGION_SELECTED );
    }
    return;
  }
}

_via_ctrl.prototype.mousedown_handler = function(e) {
  e.stopPropagation();

  var x0 = e.offsetX;
  var y0 = e.offsetY;
  this.v.last.mousedown = new _via_point(x0, y0);
  if ( this.v.state_now === this.v.state.IDLE ) {
    if ( this.v.now.meta_keypress.shiftkey ) {
      this.v.nvertex.push( x0 );
      this.v.nvertex.push( y0 );
      this.v.set_state( this.v.state.SELECT_ALL_INSIDE_AN_AREA_ONGOING );
    } else {
      // is this mousedown inside a region?
      var fileid = this.v.now.fileid;
      this.v.last.clicked_region_id = this.m.is_point_in_a_region(fileid, x0, y0);
      if ( this.v.last.clicked_region_id !== '' ) {
        // two possibilities:
        // 1. Draw region inside an existing region
        // 2. Select the region
        this.v.set_state( this.v.state.REGION_SELECT_OR_DRAW_POSSIBLE );
      } else {
        this.v.nvertex.push( x0 );
        this.v.nvertex.push( y0 );
        this.v.set_state( this.v.state.REGION_DRAW_ONGOING );
      }
    }
    return;
  }

  if ( this.v.state_now === this.v.state.REGION_DRAW_NCLICK_ONGOING ) {
    this.v.nvertex.push( x0 );
    this.v.nvertex.push( y0 );
    this.show_message( 'Press <span class="highlight">[Enter]</span> key when you finish drawing, ' +
		       '<span class="highlight">[Esc]</span> to cancel.');
    return;
  }

  if ( this.v.state_now === this.v.state.REGION_SELECTED ) {
    var fileid = this.v.now.fileid;
    var tolerance = this.v.settings.theme.MOUSE_VERTEX_TOLERANCE;
    var selected_rid_list = this.v.now.region_select.rid_list;

    var edge = this.m.is_on_these_region_edge(fileid, selected_rid_list, x0, y0, tolerance);
    if ( edge.id !== -1 ) {
      // mousedown on region edge
      this.v.last.clicked_region_id = edge.rid;
      this.v.last.clicked_region_edge_id = edge.id;
      this.v.set_state( this.v.state.REGION_RESIZE_ONGOING );
    } else {
      if ( this.v.now.meta_keypress.shiftkey ) {
        this.v.last.clicked_region_id = this.m.is_point_inside_these_regions(fileid, this.v.now.all_rid_list, x0, y0);
        if ( this.v.last.clicked_region_id !== '' ) {
          // inside a region, hence toggle region selection
          this.v.set_state( this.v.state.REGION_SELECT_TOGGLE_ONGOING );
        } else {
          // user intends to draw an area to select all regions inside the area
          this.v.set_state( this.v.state.SELECT_ALL_INSIDE_AN_AREA_ONGOING );
        }
      } else {
        this.v.last.clicked_region_id = this.m.is_point_inside_these_regions(fileid, selected_rid_list, x0, y0);
        if ( this.v.last.clicked_region_id !== '' ) {
          // mousedown inside one of the selected regions
          this.v.nvertex.push( x0 );
          this.v.nvertex.push( y0 );
          this.v.set_state( this.v.state.REGION_MOVE_ONGOING );
        } else {
          this.v.set_state( this.v.state.REGION_UNSELECT_ONGOING );
        }
      }
    }
    return;
  }
}

_via_ctrl.prototype.mouseover_handler = function(e) {
  e.stopPropagation();
  if(this.v.zoom.is_enabled) {
    this.zoom_add_panel();
  }
}

_via_ctrl.prototype.mouseout_handler = function(e) {
  e.stopPropagation();
  if(this.v.zoom.is_enabled) {
    this.zoom_remove_panel();
  }
}

_via_ctrl.prototype.touchstart_handler = function(e) {
  e.stopPropagation();
}

_via_ctrl.prototype.touchend_handler = function(e) {
  e.stopPropagation();
}

_via_ctrl.prototype.touchmove_handler = function(e) {
  e.stopPropagation();
}

///
/// Region Draw Operations
///
_via_ctrl.prototype.nvertex_clear = function() {
  this.v.nvertex.splice(0);
}

_via_ctrl.prototype.add_region_from_nvertex = function() {
  var nvertex = this.v.nvertex.slice(0);
  var shape   = this.v.now.region_shape;
  var fileid  = this.v.now.fileid;
  var tform_scale = this.v.now.tform.scale;
  var rid = this.m.region_add(fileid, tform_scale, shape, nvertex);
  // maintain a list of regions for current file
  this.v.now.all_rid_list.push(rid);
  switch(shape) {
  case this.v.settings.REGION_SHAPE.POLYGON:
  case this.v.settings.REGION_SHAPE.POLYLINE:
    // region shape requiring more than two points (polygon, polyline)
    this.v.now.polygon_rid_list.push(rid);
    break;
  default:
    // region shapes requiring just two points (rectangle, circle, etc)
    this.v.now.other_rid_list.push(rid);
    break;
  }

  this.v.last.added_region_id = rid;
  this.add_region_to_view(rid);

  this.trigger_hook(this.hook.id.REGION_ADDED, {'fileid': fileid, 'rid': rid});
  return rid;
}

_via_ctrl.prototype.remove_all_regions_from_view = function() {
  this.v.layers['rshape'].innerHTML = '';
}

_via_ctrl.prototype.add_region_to_view = function(rid) {
  var region_svg = this.m.regions[this.v.now.fileid][rid].get_svg_element();
  this.svg_apply_theme(region_svg, this.v.settings.theme.svg.REGION);
  this.v.layers['rshape'].appendChild(region_svg);

  if(this.v.zoom.is_enabled) {
    this.update_zoom_panel_contents();
  }
}

_via_ctrl.prototype.del_region_from_view = function(rid) {
  var r = this.v.layers.rshape.getElementById( rid );
  if ( r ) {
    this.v.layers.rshape.removeChild( r );
  }

  if(this.v.zoom.is_enabled) {
    this.update_zoom_panel_contents();
  }
}

_via_ctrl.prototype.update_region_in_view = function(rid) {
  var region_svg = this.v.layers.rshape.getElementById(rid);
  if ( region_svg ) {
    var new_region = this.m.regions[this.v.now.fileid][rid];
    var n = new_region.svg_attributes.length;
    for ( var i = 0; i < n; i++ ) {
      var attr = new_region.svg_attributes[i];
      region_svg.setAttributeNS(null, attr, new_region[attr]);
    }

    if(this.v.zoom.is_enabled) {
      this.update_zoom_panel_contents();
    }
  }
}

_via_ctrl.prototype.svg_apply_theme = function(svg_element, theme) {
  for ( var p in theme ) {
    if ( theme.hasOwnProperty(p) ) {
      svg_element.setAttributeNS(null, p, theme[p]);
    }
  }
}

_via_ctrl.prototype.add_tmp_region_to_view = function(nvertex_tmp,
                                                      shape,
                                                      svg_style,
                                                      rid_tmp) {
  var rid_tmp   = rid_tmp || this.v.now.view_tmp_region.DEFAULT_RID;
  var shape     = shape || this.v.now.region_shape;
  var svg_style = svg_style || this.v.settings.theme.svg.ON_DRAWING;
  var old_region_tmp = this.v.layers.rshape.getElementById( rid_tmp );
  if ( old_region_tmp ) {
    // updating existing region is more efficient than replacing it
    this.update_tmp_svg_region(old_region_tmp, nvertex_tmp);
  } else {
    var region_tmp = this.create_tmp_svg_region(shape, nvertex_tmp);
    this.svg_apply_theme(region_tmp, svg_style);

    this.v.layers.rshape.appendChild(region_tmp);
  }
}

_via_ctrl.prototype.tmp_move_selected_region_in_view = function(dx, dy) {
  var n = this.v.now.region_select.rid_list.length;
  for ( var i=0; i<n; i++ ) {
    var srid = this.v.now.region_select.rid_list[i];
    var sregion = this.m.regions[this.v.now.fileid][srid];
    var srid_tmp = this.v.now.view_tmp_region.RID_PREFIX_MOVE + srid;

    var svg_element = this.v.layers.rshape.getElementById( srid_tmp );
    if ( svg_element ) {
      // update the svg element
      this.move_tmp_svg_region( svg_element, sregion, dx, dy );
    } else {
      // create temporary region
      var new_svg_element = this.create_tmp_svg_region(sregion.shape, sregion.dview, srid_tmp);
      this.move_tmp_svg_region(new_svg_element, sregion, dx, dy);
      this.svg_apply_theme(new_svg_element, this.v.settings.theme.svg.ON_MOVE);
      this.v.layers.rshape.appendChild(new_svg_element);
    }
  }
}

_via_ctrl.prototype.tmp_resize_selected_region_in_view = function(edge_id, new_x, new_y) {
  var srid = this.v.last.clicked_region_id;
  var sregion = this.m.regions[this.v.now.fileid][srid];
  var srid_tmp = this.v.now.view_tmp_region.RID_PREFIX_RESIZE + srid;

  var svg_element = this.v.layers.rshape.getElementById( srid_tmp );
  if ( svg_element ) {
    // update the svg element
    this.resize_tmp_svg_region( svg_element, sregion, edge_id, new_x, new_y );
  } else {
    // create temporary region
    var new_svg_element = this.create_tmp_svg_region(sregion.shape, sregion.dview, srid_tmp);
    this.resize_tmp_svg_region(new_svg_element, sregion, edge_id, new_x, new_y);
    this.svg_apply_theme(new_svg_element, this.v.settings.theme.svg.ON_MOVE);
    this.v.layers.rshape.appendChild(new_svg_element);
  }
}

_via_ctrl.prototype.remove_tmp_region_from_view = function(rid) {
  var rid_tmp = rid || this.v.now.view_tmp_region.DEFAULT_RID;
  var region_tmp = this.v.layers.rshape.getElementById( rid_tmp );
  if ( region_tmp ) {
    this.v.layers.rshape.removeChild( region_tmp );
  }
}

///
/// Region management
///
_via_ctrl.prototype.delete_selected_regions = function() {
  this.region_delete( this.v.now.region_select.rid_list );
  this.v.now.region_select.rid_list = [];
  this.v.now.region_select.fileid = '';
  this.v.set_state( this.v.state.IDLE );
}

_via_ctrl.prototype.region_delete = function(rid_list) {
  var n = rid_list.length;
  var fileid = this.v.now.fileid;
  for ( var i=0; i<n; i++ ) {
    var rid = rid_list[i];
    var shape = this.m.regions[fileid][rid].shape;
    delete this.m.regions[fileid][rid];
    console.log('deleting region ' + rid);

    this.v.now.all_rid_list.splice( this.v.now.all_rid_list.indexOf(rid), 1 );
    if ( shape === this.v.settings.REGION_SHAPE.POLYGON ) {
      this.v.now.polygon_rid_list.splice( this.v.now.polygon_rid_list.indexOf(rid), 1 );
    } else {
      this.v.now.other_rid_list.splice( this.v.now.other_rid_list.indexOf(rid), 1 );
    }

    // delete region from view
    this.del_region_from_view(rid);
  }

  if(this.v.zoom.is_enabled) {
    this.update_zoom_panel_contents();
  }
}

_via_ctrl.prototype.region_delete_all = function() {
  this.m.regions[this.v.now.fileid] = [];
  this.v.now.all_rid_list = [];
  this.v.now.polygon_rid_list = [];
  this.v.now.other_rid_list = [];
  this.remove_all_regions_from_view();
}

///
/// maintainers for 'now' : the current file being annotated
///
_via_ctrl.prototype.set_now_file = function( fileid ) {
  return new Promise( function(ok_callback, err_callback) {
    var filereader = new FileReader();
    filereader.addEventListener( 'load', function() {
      this.v.now.fileid = fileid;

      switch( this.m.files.metadata[fileid].type ) {
      case 'image':
        this.v.now.content = document.createElement('img');
        this.v.now.content.setAttribute('id', fileid);
        this.v.now.content.addEventListener('load', function() {
          this.v.now.content_metadata.type = 'image';
          this.v.now.content_metadata.is_paused = false;
          this.v.layers['top'].style.pointerEvents = 'auto'; // let via handle the mouse events on image
	  ok_callback();
        }.bind(this), false);

        break;
      case 'video':
        this.v.now.content = document.createElement('video');
        this.v.now.content.setAttribute('id', fileid);
        this.v.now.content.setAttribute('autoplay', 'true');
        this.v.now.content.setAttribute('loop', 'true');
        this.v.now.content.setAttribute('controls', '');
        this.v.now.content.addEventListener('canplay', function() {
          this.v.now.content_metadata.type = 'video';
          this.v.now.content_metadata.is_paused = false;
	  ok_callback();
        }.bind(this), false);

        this.v.now.content.addEventListener('pause', function(e) {
          this.v.now.content_metadata.is_paused = true;
          this.v.layers['top'].style.pointerEvents = 'auto';
          this.trigger_hook(this.hook.id.VIDEO_ON_PAUSE, {'fileid': this.v.now.fileid, 'time': this.v.now.content.currentTime} );
          //this.show_message('Video paused, click again to play video. Draw a region by dragging mouse button while keeping it pressed.');
        }.bind(this));

        this.v.now.content.addEventListener('seeking', function(e) {
          this.trigger_hook(this.hook.id.VIDEO_ON_SEEK, {'fileid': this.v.now.fileid, 'time': this.v.now.content.currentTime} );
        }.bind(this));

        this.v.now.content.addEventListener('play', function(e) {
          this.v.now.content_metadata.is_paused = false;
          this.v.layers['top'].style.pointerEvents = 'none'; // let video handler handler the mouse events
          this.trigger_hook(this.hook.id.VIDEO_ON_PLAY, {'fileid': this.v.now.fileid, 'time': this.v.now.content.currentTime} );
        }.bind(this));

        this.v.now.content.addEventListener('abort', function(e) {
          console.log('Video abort event raised!');
          console.log(e);
        }.bind(this));

        break;

      default:
        err_callback('Unknown content type [' + this.m.files.metadata[fileid].type +
                     '] for file : ' + this.m.files.metadata[this.v.now.fileid].filename);
      }
      this.v.now.content.addEventListener( 'error', err_callback);
      this.v.now.content.addEventListener( 'abort', err_callback);

      this.v.now.content.src = filereader.result;
    }.bind(this));

    filereader.addEventListener( 'error', err_callback);
    filereader.addEventListener( 'abort', err_callback);

    if ( this.m.files.metadata[fileid].source === 'local' ) {
      filereader.readAsDataURL( this.m.files.content[fileid] );
    } else {
      filereader.readAsText( new Blob([this.m.files.content[fileid]]) );
    }
  }.bind(this));
}

///
/// utility functions
///
_via_ctrl.prototype.file_hash = function(filename, filesize, frame, count) {
  var fileid_str = filename + (filesize || -1 ) + (frame || 0) + (count || 1);

  // @@todo: fixme
  // avoid crypto.subtle.digest() because it is not allowed over http:// connection by chrome
  //return this.hash( fileid_str );

  return new Promise( function(ok_callback, err_callback) {
    ok_callback(fileid_str);
  });
}

_via_ctrl.prototype.url_filetype = function(url) {
  var dot_index = url.lastIndexOf('.');
  var ext = url.substr(dot_index);

  var type = '';
  switch( ext.toLowerCase() ) {
  case '.webm':
  case '.ogg':
  case '.mp4':
  case '.ogv':
    type = 'video';
    break;
  case '.jpg':
  case '.png':
  case '.jpeg':
  case '.svg':
    type = 'image'
    break;

  default:
    type = 'unknown';
  }
  return type;
}

_via_ctrl.prototype.are_mouse_events_nearby = function(p0, p1) {
  var dx = p1.x - p0.x;
  var dy = p1.y - p0.y;
  if ( ( dx*dx + dy*dy ) <= this.v.settings.theme.MOUSE_CLICK_TOLERANCE2 ) {
    return true;
  } else {
    return false;
  }
}

// to draw svg shape as the region is being drawn, moved or resized by the user
_via_ctrl.prototype.create_tmp_svg_region = function(shape, nvertex, rid_tmp) {
  var _VIA_SVG_NS = 'http://www.w3.org/2000/svg';
  var svg_element = document.createElementNS(_VIA_SVG_NS, shape);
  var rid_tmp = rid_tmp || this.v.now.view_tmp_region.DEFAULT_RID;

  svg_element.setAttributeNS(null, 'id', rid_tmp);

  var dx = Math.abs(nvertex[2] - nvertex[0]);
  var dy = Math.abs(nvertex[3] - nvertex[1]);

  switch(shape) {
  case this.v.settings.REGION_SHAPE.RECT:
    // ensure that (x0,y0) corresponds to top-left corner of rectangle
    var x0 = nvertex[0];
    var x1 = nvertex[2];
    var y0 = nvertex[1];
    var y1 = nvertex[3];
    if ( nvertex[0] > nvertex[2] ) {
      x0 = nvertex[2];
      x1 = nvertex[0];
    }
    if ( nvertex[1] > nvertex[3] ) {
      y0 = nvertex[3];
      y1 = nvertex[1];
    }

    svg_element.setAttributeNS(null, 'x', x0);
    svg_element.setAttributeNS(null, 'y', y0);
    svg_element.setAttributeNS(null, 'width',  x1 - x0);
    svg_element.setAttributeNS(null, 'height', y1 - y0);
    break;
  case this.v.settings.REGION_SHAPE.CIRCLE:
    var r  = Math.round( Math.sqrt( dx * dx + dy * dy ) );
    svg_element.setAttributeNS(null, 'cx', nvertex[0]);
    svg_element.setAttributeNS(null, 'cy', nvertex[1]);
    svg_element.setAttributeNS(null, 'r', r);
    break;
  case this.v.settings.REGION_SHAPE.ELLIPSE:
    svg_element.setAttributeNS(null, 'cx', nvertex[0]);
    svg_element.setAttributeNS(null, 'cy', nvertex[1]);
    svg_element.setAttributeNS(null, 'rx', dx);
    svg_element.setAttributeNS(null, 'ry', dy);
    break;
  case this.v.settings.REGION_SHAPE.LINE:
    // preserve the start(x1,y1) and end(x2,y2) points as drawn by user
    svg_element.setAttributeNS(null, 'x1', nvertex[0]);
    svg_element.setAttributeNS(null, 'y1', nvertex[1]);
    svg_element.setAttributeNS(null, 'x2', nvertex[2]);
    svg_element.setAttributeNS(null, 'y2', nvertex[3]);
    break;
  case this.v.settings.REGION_SHAPE.POLYLINE:
  case this.v.settings.REGION_SHAPE.POLYGON:
    var n = nvertex.length;
    var points = [];
    for ( var i = 0; i < n; i += 2 ) {
      points.push(nvertex[i] + ' ' + nvertex[i+1]);
    }
    svg_element.setAttributeNS(null, 'points', points.join(','));
    break;
  case this.v.settings.REGION_SHAPE.POINT:
    svg_element.setAttributeNS(null, 'cx', nvertex[0]);
    svg_element.setAttributeNS(null, 'cy', nvertex[1]);
    svg_element.setAttributeNS(null, 'r', this.v.settings.theme.svg.POINT_SHAPE_DRAW_RADIUS);
    break;
  }
  return svg_element;
}

_via_ctrl.prototype.update_tmp_svg_region = function(svg_element, nvertex) {
  var dx = Math.abs(nvertex[2] - nvertex[0]);
  var dy = Math.abs(nvertex[3] - nvertex[1]);

  switch( svg_element.localName ) {
  case this.v.settings.REGION_SHAPE.RECT:
    // ensure that (x0,y0) corresponds to top-left corner of rectangle
    var x0 = nvertex[0];
    var x1 = nvertex[2];
    var y0 = nvertex[1];
    var y1 = nvertex[3];
    if ( nvertex[0] > nvertex[2] ) {
      x0 = nvertex[2];
      x1 = nvertex[0];
    }
    if ( nvertex[1] > nvertex[3] ) {
      y0 = nvertex[3];
      y1 = nvertex[1];
    }

    svg_element.setAttributeNS(null, 'x', x0);
    svg_element.setAttributeNS(null, 'y', y0);
    svg_element.setAttributeNS(null, 'width',  x1 - x0);
    svg_element.setAttributeNS(null, 'height', y1 - y0);
    break;
  case this.v.settings.REGION_SHAPE.CIRCLE:
    var r  = Math.round( Math.sqrt( dx * dx + dy * dy ) );
    svg_element.setAttributeNS(null, 'r', r);
    break;
  case this.v.settings.REGION_SHAPE.ELLIPSE:
    svg_element.setAttributeNS(null, 'rx', dx);
    svg_element.setAttributeNS(null, 'ry', dy);
    break;
  case this.v.settings.REGION_SHAPE.LINE:
    // preserve the start(x1,y1) and end(x2,y2) points as drawn by user
    svg_element.setAttributeNS(null, 'x2', nvertex[2]);
    svg_element.setAttributeNS(null, 'y2', nvertex[3]);
    break;
  case this.v.settings.REGION_SHAPE.POLYLINE:
  case this.v.settings.REGION_SHAPE.POLYGON:
    var points = svg_element.getAttributeNS(null, 'points');
    var n = nvertex.length;
    var points = [];
    for ( var i = 0; i < n; i += 2 ) {
      points.push( nvertex[i] + ' ' + nvertex[i+1] );
    }
    svg_element.setAttributeNS(null, 'points', points.join(','));
    break;
  case this.v.settings.REGION_SHAPE.POINT:
    svg_element.setAttributeNS(null, 'cx', nvertex[0]);
    svg_element.setAttributeNS(null, 'cy', nvertex[1]);
    svg_element.setAttributeNS(null, 'r', this.v.settings.theme.svg.POINT_SHAPE_DRAW_RADIUS);
    break;
  }
}

_via_ctrl.prototype.move_tmp_svg_region = function(svg_element, region, dx, dy) {

  switch( svg_element.localName ) {
  case this.v.settings.REGION_SHAPE.RECT:
    svg_element.setAttributeNS(null, 'x', region.x + dx);
    svg_element.setAttributeNS(null, 'y', region.y + dy);
    break;
  case this.v.settings.REGION_SHAPE.CIRCLE:
  case this.v.settings.REGION_SHAPE.ELLIPSE:
  case this.v.settings.REGION_SHAPE.POINT:
    svg_element.setAttributeNS(null, 'cx', region.cx + dx);
    svg_element.setAttributeNS(null, 'cy', region.cy + dy);
    break;
  case this.v.settings.REGION_SHAPE.LINE:
    svg_element.setAttributeNS(null, 'x1', region.x1 + dx);
    svg_element.setAttributeNS(null, 'y1', region.y1 + dy);
    svg_element.setAttributeNS(null, 'x2', region.x2 + dx);
    svg_element.setAttributeNS(null, 'y2', region.y2 + dy);
    break;
  case this.v.settings.REGION_SHAPE.POLYLINE:
  case this.v.settings.REGION_SHAPE.POLYGON:
    var points = [];
    var n = region.dview.length;
    for ( var i = 0; i < n; i += 2 ) {
      points.push( (region.dview[i] + dx) + ' ' + (region.dview[i+1] + dy) );
    }
    svg_element.setAttributeNS(null, 'points', points.join(','));
    break;
  }
}

_via_ctrl.prototype.resize_tmp_svg_region = function(svg_element, region, edge_id, new_x, new_y, rid_tmp) {

  // WARNING: do not invoke resize in the original region!
  // create a copy of region
  var rid_tmp = rid_tmp || this.v.now.view_tmp_region.DEFAULT_RID;
  var r = new _via_region(region.shape,
                          rid_tmp,
                          region.dview,
                          1);
  r.resize(edge_id, new_x, new_y);

  var new_svg_element = r.get_svg_element();
  var n = r.svg_attributes.length;
  for (var i=0; i<n; i++ ) {
    var p = r.svg_attributes[i];
    var old_value = svg_element.getAttributeNS(null, p);
    var new_value = new_svg_element.getAttributeNS(null, p);
    if ( new_value !== old_value ) {
      svg_element.setAttributeNS(null, p, new_value);
    }
  }
}

///
/// Zoom support
///
_via_ctrl.prototype.zoom_remove_panel = function() {
  this.v.layers['zoom'].innerHTML = '';
}
_via_ctrl.prototype.zoom_add_panel = function() {
  this.v.zoom.container = document.createElement('div');
  this.v.zoom.container.setAttribute('id', this.v.zoom.container_id);

  // add filecontent
  this.v.zoom.filecontent = this.v.layers['filecontent'].cloneNode(true);
  this.v.zoom.filecontent.setAttribute('id', 'zoom_filecontent');
  this.v.zoom.filecontent.removeAttribute('style');
  this.v.zoom.filecontent.childNodes[0].removeAttribute('id');
  this.v.zoom.filecontent.childNodes[0].removeAttribute('style');

  var fcw = this.v.now.tform.width * this.v.zoom.scale;
  var fch = this.v.now.tform.height * this.v.zoom.scale;
  this.v.zoom.filecontent.childNodes[0].setAttribute('width', fcw);
  this.v.zoom.filecontent.childNodes[0].setAttribute('height', fch);

  this.v.layers['zoom'].appendChild(this.v.zoom.container);
}
_via_ctrl.prototype.zoom_activate = function() {
  if ( this.v.zoom.is_enabled ) {
    // change zoom scale if already activated
    this.v.zoom.scale += 0.5;
  } else {
    this.v.zoom.is_enabled = true;
  }
  this.show_message('Zoom <span class="highlight">' + this.v.zoom.scale + 'X</span> enabled');
  // handled by mouseover_handler()
  //this.zoom_add_panel();
}

_via_ctrl.prototype.zoom_deactivate = function() {
  this.v.zoom.is_enabled = false;
  this.v.zoom.scale = this.v.zoom.DEFAULT_SCALE;
  this.show_message('Zoom disabled');
  // handled by mouseout_handler()
  //this.zoom_remove_panel();
}

_via_ctrl.prototype.update_zoom_panel_contents = function() {
  var px = this.v.last.mousemove.x;
  var py = this.v.last.mousemove.y;

  // position zoom panel
  var zoom_panel_left = px - this.v.zoom.sizeby2;
  var zoom_panel_top  = py - this.v.zoom.sizeby2;
  var style = [];
  style.push('position: absolute');
  style.push('overflow: hidden'); // without this, zoom will break
  style.push('width:' + this.v.zoom.size + 'px');
  style.push('height:' + this.v.zoom.size + 'px');
  style.push('top:' + zoom_panel_top + 'px');
  style.push('left:' + zoom_panel_left + 'px');
  style.push('border: 1px solid red');
  style.push('border-radius:' + this.v.zoom.sizeby2 + 'px');
  this.v.zoom.container.setAttribute('style', style.join(';'));

  // position filecontent
  style = [];
  style.push('position: absolute');
  var scaled_img_left = this.v.zoom.sizeby2 - px * this.v.zoom.scale;
  var scaled_img_top  = this.v.zoom.sizeby2 - py * this.v.zoom.scale;
  style.push('top:' + scaled_img_top + 'px');
  style.push('left:' + scaled_img_left + 'px');
  this.v.zoom.filecontent.childNodes[0].setAttribute('style', style.join(';'));

  // position rshape
  var rshape = this.v.layers['rshape'].cloneNode(true);
  rshape.setAttribute('id', 'zoom_rshape');
  for(var i=0; i<rshape.childNodes.length; i++) {
    rshape.childNodes[i].removeAttribute('id');
  }
  var fcw = this.v.now.tform.width * this.v.zoom.scale;
  var fch = this.v.now.tform.height * this.v.zoom.scale;
  var vbox = [0, 0, this.v.now.tform.width, this.v.now.tform.height];
  rshape.setAttribute('viewBox', vbox.join(' '));
  rshape.setAttribute('width', fcw);
  rshape.setAttribute('height', fch);

  rshape.removeAttribute('style');
  style = [];
  style.push('position: absolute');
  var scaled_img_left = this.v.zoom.sizeby2 - px * this.v.zoom.scale;
  var scaled_img_top  = this.v.zoom.sizeby2 - py * this.v.zoom.scale;
  style.push('top:' + scaled_img_top + 'px');
  style.push('left:' + scaled_img_left + 'px');
  rshape.setAttribute('style', style.join(';'));

  if ( this.v.zoom.container.childNodes.length === 0 ) {
    // first time: add both filecontent and region shapes
    this.v.zoom.container.appendChild(this.v.zoom.filecontent);
    this.v.zoom.container.appendChild(rshape);
  } else {
    // next time: just update region shapes
    this.v.zoom.container.replaceChild(rshape, this.v.zoom.container.childNodes[1]);
  }
}


///
/// Region select/unslect handler
///
_via_ctrl.prototype.region_select_next = function() {

}

_via_ctrl.prototype.region_select_prev = function() {

}

_via_ctrl.prototype.region_select_toggle = function(rid) {
  if ( this.v.now.region_select.rid_list.includes(rid) ) {
    this.region_unselect(rid);
  } else {
    this.region_select(rid);
  }
}

_via_ctrl.prototype.region_select = function(rid) {
  var new_svg_element = this.v.layers.rshape.getElementById(rid);
  this.svg_apply_theme( new_svg_element, this.v.settings.theme.svg.SELECTED_REGION );
  this.v.now.region_select.rid_list.push(rid);
}

_via_ctrl.prototype.region_select_all = function() {
  this.region_select_n( this.v.now.all_rid_list );
  if ( this.v.now.region_select.rid_list.length ) {
    this.v.set_state( this.v.state.REGION_SELECTED );
  }
}

_via_ctrl.prototype.region_select_n = function(rid_list) {
  var n = rid_list.length;
  for ( var i=0; i<n; i++ ) {
    this.region_select( rid_list[i] );
  }
}

_via_ctrl.prototype.region_unselect = function(rid) {
  var index = this.v.now.region_select.rid_list.indexOf(rid);
  this.v.now.region_select.rid_list.splice(index, 1);

  var old_svg_element = this.v.layers.rshape.getElementById( rid );
  this.svg_apply_theme( old_svg_element, this.v.settings.theme.svg.REGION );
}

_via_ctrl.prototype.region_unselect_all = function() {
  var n = this.v.now.region_select.rid_list.length;
  for ( var i=0; i<n; i++ ) {
    var rid = this.v.now.region_select.rid_list[i];
    var old_svg_element = this.v.layers.rshape.getElementById( rid );
    this.svg_apply_theme( old_svg_element, this.v.settings.theme.svg.REGION );
  }
  this.v.now.region_select.rid_list = [];
  this.v.now.region_select.fileid = '';
}

_via_ctrl.prototype.get_regions_inside_an_area = function(x0, y0, x1, y1) {
  var regions = [];
  var area = new _via_region(this.v.settings.REGION_SHAPE.RECT,
                             '',
                             [x0, y0, x1, y1], 1);

  var n = this.v.now.all_rid_list.length;
  var fileid = this.v.now.fileid;
  for ( var i=0; i<n; i++ ) {
    var rid = this.v.now.all_rid_list[i];
    var r = this.m.regions[fileid][rid];
    var nd = r.dview.length;

    for ( var j=0; j<nd; j += 2 ) {
      if ( area.is_inside( r.dview[j], r.dview[j+1] ) ) {
        regions.push(rid);
        break;
      }
    }
  }
  return regions;
}

////////////////////////////////////////////////////////////////////////////////
//
// @file        _via.js
// @description VGG Image Annotator (VIA) application entrypoint
// @author      Abhishek Dutta <adutta@robots.ox.ac.uk>
// @date        17 June 2017
//
////////////////////////////////////////////////////////////////////////////////

function _via() {
  this.m = new _via_model();  // model
  this.v = new _via_view();   // view
  this.c = new _via_ctrl();   // controller

  this.init = function(view_panel, message_panel) {
    console.log("Initializing via_ng ...");
    this.m.init( this.c );
    this.v.init( this.c, view_panel, message_panel );
    this.c.init( this.m, this.v );
  }
}

_via.prototype.register_ui_action_handler = function(html_element) {
  var p = document.getElementById(html_element);
  if(p) {
    p.addEventListener("click", this.v, false);
  }
}
