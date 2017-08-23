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

// you can customize your own columns ...

$col = array();
$col["title"] = "Id"; // caption of column
$col["name"] = "id"; 
$col["width"] = "10";
# $col["hidden"] = true; // hide column by default
$cols[] = $col;		

$col = array();
$col["title"] = "Client";
$col["name"] = "name";
$col["width"] = "100";
$col["editable"] = false; // this column is not editable
$col["align"] = "center"; // this column is not editable
$col["search"] = false; // this column is not searchable

# $col["formatter"] = "image"; // format as image -- if data is image url e.g. http://<domain>/test.jpg
# $col["formatoptions"] = array("width"=>'20',"height"=>'30'); // image width / height etc

$cols[] = $col;

$col = array();
$col["title"] = "Note";
$col["name"] = "note";
# $col["width"] = "300"; // not specifying width will expand to fill space
$col["sortable"] = false; // this column is not sortable
$col["search"] = false; // this column is not searchable
$col["editable"] = true;
$col["edittype"] = "textarea"; // render as textarea on edit
$col["editoptions"] = array("rows"=>2, "cols"=>20); // with these attributes
$cols[] = $col;

$col = array();
$col["title"] = "Date";
// fieldname is not with tablealias in sql, so we used plain fieldname
$col["name"] = "invdate"; 
$col["width"] = "50";
$col["editable"] = true; // this column is editable
$col["editoptions"] = array("size"=>20); // with default display of textbox with size 20
$col["editrules"] = array("required"=>true); // and is required
$col["formatter"] = "date"; // format as date
$cols[] = $col;
		
		
$col = array();
$col["title"] = "Total";
$col["name"] = "total";
$col["width"] = "50";
$col["editable"] = true;
$col["summaryType"] = "sum"; // available grouping fx: sum, count, min, max
$col["summaryTpl"] = '<b>Total: ${0}</b>'; // display html for summary row - work when "groupSummary" is set true. search below

// default render is textbox
$col["editoptions"] = array("value"=>'10');

// can be switched to select (dropdown)
# $col["edittype"] = "select"; // render as select
# $col["editoptions"] = array("value"=>'10:$10;20:$20;30:$30;40:$40;50:$50'); // with these values "key:value;key:value;key:value"

$cols[] = $col;

$col = array();
$col["title"] = "Closed";
$col["name"] = "closed";
$col["width"] = "50";
$col["editable"] = true;
$col["edittype"] = "checkbox"; // render as checkbox
$col["editoptions"] = array("value"=>"Yes:No"); // with these values "checked_value:unchecked_value"
$cols[] = $col;

$g = new jqgrid();

// $grid["url"] = ""; // your paramterized URL -- defaults to REQUEST_URI
$grid["rowNum"] = 10; // by default 20
$grid["sortname"] = 'id'; // by default sort grid by this field
$grid["sortorder"] = "desc"; // ASC or DESC
$grid["caption"] = "Invoice Data"; // caption of grid
$grid["autowidth"] = true; // expand grid to screen width
$grid["multiselect"] = true; // allow you to multi-select through checkboxes

$grid["grouping"] = true; // 
$grid["groupingView"] = array();
$grid["groupingView"]["groupField"] = array("closed"); // specify column name to group listing
$grid["groupingView"]["groupColumnShow"] = array(false); // either show grouped column in list or not (default: true)
$grid["groupingView"]["groupText"] = array("<b>{0} - {1} Item(s)</b>"); // {0} is grouped value, {1} is count in group
$grid["groupingView"]["groupOrder"] = array("asc"); // show group in asc or desc order
$grid["groupingView"]["groupDataSorted"] = array(true); // show sorted data within group
$grid["groupingView"]["groupSummary"] = array(true); // work with summaryType, summaryTpl, see column: $col["name"] = "total";
$grid["groupingView"]["groupCollapse"] = false; // Turn true to show group collapse (default: false) 
$grid["groupingView"]["showSummaryOnHide"] = true; // show summary row even if group collapsed (hide) 

// export XLS file
// export to excel parameters
$grid["export"] = array("format"=>"xlsx", "filename"=>"my-file", "sheetname"=>"test");

// RTL support
// $grid["direction"] = "rtl";

// $grid["cellEdit"] = true; // inline cell editing, like spreadsheet

$g->set_options($grid);

$g->set_actions(array(	
						"add"=>true, // allow/disallow add
						"edit"=>true, // allow/disallow edit
						"delete"=>true, // allow/disallow delete
						"rowactions"=>true, // show/hide row wise edit/del/save option
						"export"=>true, // show/hide export to excel option
						"autofilter" => true, // show/hide autofilter for search
						"search" => "advance" // show single/multi field search condition (e.g. simple or advance)
					) 
				);

// you can provide custom SQL query to display data
$g->select_command = "SELECT * FROM (SELECT i.id, invdate , c.firstname,
						i.note, i.total, i.closed FROM invheader i
						INNER JOIN clients c ON c.client_id = i.client_id) o";

// this db table will be used for add,edit,delete
$g->table = "invheader";

// pass the cooked columns to grid
$g->set_columns($cols);

$grid_id = "list1";
// generate grid output, with unique grid name as 'list1'
$out = $g->render($grid_id);

$themes = array("redmond","smoothness","start","dot-luv","excite-bike","flick","ui-darkness","cupertino","dark-hive");
$i = rand(0,8);
?>
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.1//EN" "http://www.w3.org/TR/xhtml11/DTD/xhtml11.dtd">
<html>
<head>
	<link rel="stylesheet" type="text/css" media="screen" href="js/themes/<?php echo $themes[0]?>/jquery-ui.custom.css"></link>	
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
	<br>
	Dynamic Group By: 
	<select id="chngroup">
		<?php foreach($cols as $c) { ?>
		<option value="<?php echo $c["name"] ?>"><?php echo $c["title"] ?></option>
		<?php } ?>
		<option value="clear">Clear</option> 
	</select>
	<script>
		jQuery("#chngroup").change(function()
		{
			var vl = jQuery(this).val(); 
			if(vl) 
			{ 
				if(vl == "clear") 
					jQuery("#<?php echo $grid_id ?>").jqGrid('groupingRemove',true); 
				else 
					jQuery("#<?php echo $grid_id ?>").jqGrid('groupingGroupBy',vl); 
			} 
		});
	</script>	
	</div>	
</body>
</html>