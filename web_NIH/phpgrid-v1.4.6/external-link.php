<?php 
/**
 * PHP Grid Component
 *
 * @author Abu Ghufran <gridphp@gmail.com> - http://www.phpgrid.org
 * @version 1.4.6
 * @license: see license.txt included in package
 */
 
$conn = mysql_connect(“ip”, “user”, “pass”);
mysql_select_db(“db_name”);

include("inc/jqgrid_dist.php");

$g = new jqgrid();

$col = array();
$col["title"] = "Id"; // caption of column
$col["name"] = "id"; 
$col["width"] = "10";
$cols[] = $col;		

		
$col = array();
$col["title"] = "Client";
$col["name"] = "lastname";
$col["width"] = "100";
$col["search"] = true;
$col["editable"] = false;
$col["export"] = false; // this column will not be exported
$col["link"] = "http://www.rcalix.com/clients/gridsexample/phpgrid-v1.4.6/external-link.php?id={id}"; // e.g. http://domain.com?id={id} given that, there is a column with $col["name"] = "id" exist
$col["linkoptions"] = "target='_blank'"; // extra params with <a> tag

$cols[] = $col;

///clients/gridsexample/phpgrid-v1.4.6/external-link.php
$col = array();
$col["title"] = "Date";
$col["name"] = "invdate"; 
$col["width"] = "50";
$col["editable"] = true; // this column is editable
$col["editoptions"] = array("size"=>20); // with default display of textbox with size 20
$col["editrules"] = array("required"=>true); // and is required
$col["formatter"] = "date"; // format as date
$col["formatoptions"] = array("srcformat"=>'Y-m-d',"newformat"=>'d/m/Y'); // @todo: format as date, not working with editing
$col["search"] = false;
$cols[] = $col;

$grid["sortname"] = 'id'; // by default sort grid by this field
$grid["sortorder"] = "desc"; // ASC or DESC
$grid["caption"] = "Invoice Data"; // caption of grid
$grid["autowidth"] = true; // expand grid to screen width
$grid["multiselect"] = false; // allow you to multi-select through checkboxes

// export XLS file
// export to excel parameters - range could be "all" or "filtered"
$grid["export"] = array("format"=>"xlsx", "filename"=>"my-file", "sheetname"=>"test");


// export PDF file
// export to excel parameters
$grid["export"] = array("format"=>"pdf", "filename"=>"my-file", "heading"=>"Invoice Details", "orientation"=>"landscape");

// export filtered data or all data
$grid["export"]["range"] = "filtered"; // or "all"

$g->set_options($grid);

$g->set_actions(array(	
						"add"=>false, // allow/disallow add
						"edit"=>false, // allow/disallow edit
						"delete"=>true, // allow/disallow delete
						"rowactions"=>true, // show/hide row wise edit/del/save option
						"export"=>true, // show/hide export to excel option
						"autofilter" => true, // show/hide autofilter for search
						"search" => "advance" // show single/multi field search condition (e.g. simple or advance)
					) 
				);

// you can provide custom SQL query to display data
$g->select_command = "SELECT * FROM (SELECT i.id, invdate , c.lastname FROM invheader i
						INNER JOIN clients c ON c.client_id = i.client_id) o";

// this db table will be used for add,edit,delete
//$g->table = "invheader";

// pass the cooked columns to grid
$g->set_columns($cols);

// generate grid output, with unique grid name as 'list1'
$out = $g->render("list1");
?>
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.1//EN" "http://www.w3.org/TR/xhtml11/DTD/xhtml11.dtd">
<html>
<head>
	<link rel="stylesheet" type="text/css" media="screen" href="js/themes/smoothness/jquery-ui.custom.css"></link>	
	<link rel="stylesheet" type="text/css" media="screen" href="js/jqgrid/css/ui.jqgrid.css"></link>	
	
	<script src="js/jquery.min.js" type="text/javascript"></script>
	<script src="js/jqgrid/js/i18n/grid.locale-en.js" type="text/javascript"></script>
	<script src="js/jqgrid/js/jquery.jqGrid.min.js" type="text/javascript"></script>	
	<script src="js/themes/jquery-ui.custom.min.js" type="text/javascript"></script>
</head>
<body>
	<div style="margin:10px">
	<br>
	<?php echo $out?>
	</div>
</body>
</html>