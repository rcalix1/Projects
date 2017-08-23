<?php 
/**
 * PHP Grid Component
 *
 * @author Abu Ghufran <gridphp@gmail.com> - http://www.phpgrid.org
 * @version 1.4.6
 * @license: see license.txt included in package
 */

$conn = mysql_connect(ÒipÓ, Òuser", ÒpassÓ);
mysql_select_db(Òdb_nameÓ);

// set your db encoding -- for ascent chars (if required)
mysql_query("SET NAMES 'utf8'");

include("inc/jqgrid_dist.php");

// you can customize your own columns ...

$col = array();
$col["title"] = "Id"; // caption of column
$col["name"] = "id"; // grid column name, must be exactly same as returned column-name from sql (tablefield or field-alias) 
$col["width"] = "10";
$cols[] = $col;		


$col = array();
$col["title"] = "medicine";
$col["name"] = "medicine";
$col["width"] = "20";
$col["editable"] = false;
$col["edittype"] = "checkbox"; // render as checkbox
$col["editoptions"] = array("value"=>"1:0"); // with these values "checked_value:unchecked_value"
#$col["edittype"] = "select"; // render as select
#$col["editoptions"] = array("value"=>'No:Not Booked eg. Ñ, Í,É;Yes:Yes it is Booked eg. Ñ, Í,É'); 
$cols[] = $col;

$col = array();
$col["title"] = "effect_candidate";
$col["name"] = "effect_cand";
$col["width"] = "40"; // not specifying width will expand to fill space
$col["sortable"] = true; // this column is not sortable
$col["search"] = true; // this column is not searchable
$col["editable"] = false;
//$col["editoptions"] = array("rows"=>2, "cols"=>20); // with these attributes
$cols[] = $col;

$col = array();
$col["title"] = "effect_hotness";
$col["name"] = "effect_hotness";
$col["width"] = "35";
$col["editable"] = false;
// To mask password field, apply following attribs
# $col["edittype"] = "password";
# $col["formatter"] = "password";
// default render is textbox
// $col["editoptions"] = array("value"=>'10');
// can be switched to select (dropdown)
# $col["edittype"] = "select"; // render as select
# $col["editoptions"] = array("value"=>'10:$10;20:$20;30:$30;40:$40;50:$50'); // with these values "key:value;key:value;key:value"

$cols[] = $col;



$col = array();
$col["title"] = "effect_root_synonym";
$col["name"] = "root_synonym";
$col["width"] = "45";
$col["editable"] = false;
// default render is textbox
// $col["editoptions"] = array("value"=>'10');
// can be switched to select (dropdown)
# $col["edittype"] = "select"; // render as select
# $col["editoptions"] = array("value"=>'10:$10;20:$20;30:$30;40:$40;50:$50'); // with these values "key:value;key:value;key:value"

$cols[] = $col;


$col = array();
$col["title"] = "stem";
$col["name"] = "stem";
$col["width"] = "20";
$col["editable"] = false;
// default render is textbox
// $col["editoptions"] = array("value"=>'10');
// can be switched to select (dropdown)
# $col["edittype"] = "select"; // render as select
# $col["editoptions"] = array("value"=>'10:$10;20:$20;30:$30;40:$40;50:$50'); // with these values "key:value;key:value;key:value"

$cols[] = $col;



$col = array();
$col["title"] = "pos";
$col["name"] = "sent_pos";
$col["width"] = "15";
$col["editable"] = false;
//$col["edittype"] = "checkbox"; // render as checkbox
//$col["editoptions"] = array("value"=>"1:0"); // with these values "checked_value:unchecked_value"
#$col["edittype"] = "select"; // render as select
#$col["editoptions"] = array("value"=>'No:Not Booked eg. Ñ, Í,É;Yes:Yes it is Booked eg. Ñ, Í,É'); 
$cols[] = $col;



$col = array();
$col["title"] = "neg";
$col["name"] = "sent_neg";
$col["width"] = "15";
$col["editable"] = false;
//$col["edittype"] = "checkbox"; // render as checkbox
//$col["editoptions"] = array("value"=>"1:0"); // with these values "checked_value:unchecked_value"
#$col["edittype"] = "select"; // render as select
#$col["editoptions"] = array("value"=>'No:Not Booked eg. Ñ, Í,É;Yes:Yes it is Booked eg. Ñ, Í,É'); 
$cols[] = $col;

$col = array();
$col["title"] = "neu";
$col["name"] = "sent_neu";
$col["width"] = "15";
$col["editable"] = false;
//$col["edittype"] = "checkbox"; // render as checkbox
//$col["editoptions"] = array("value"=>"1:0"); // with these values "checked_value:unchecked_value"
#$col["edittype"] = "select"; // render as select
#$col["editoptions"] = array("value"=>'No:Not Booked eg. Ñ, Í,É;Yes:Yes it is Booked eg. Ñ, Í,É'); 
$cols[] = $col;

$col = array();
$col["title"] = "comp";
$col["name"] = "sent_comp";
$col["width"] = "20";
$col["editable"] = false;
//$col["edittype"] = "checkbox"; // render as checkbox
//$col["editoptions"] = array("value"=>"1:0"); // with these values "checked_value:unchecked_value"
#$col["edittype"] = "select"; // render as select
#$col["editoptions"] = array("value"=>'No:Not Booked eg. Ñ, Í,É;Yes:Yes it is Booked eg. Ñ, Í,É'); 
$cols[] = $col;


$col = array();
$col["title"] = "tweet_id";
$col["name"] = "tweet_id";
$col["width"] = "40";
$col["editable"] = false;
// default render is textbox
// $col["editoptions"] = array("value"=>'10');
// can be switched to select (dropdown)
# $col["edittype"] = "select"; // render as select
# $col["editoptions"] = array("value"=>'10:$10;20:$20;30:$30;40:$40;50:$50'); // with these values "key:value;key:value;key:value"

$cols[] = $col;

$col = array();
$col["title"] = "tweet";
$col["name"] = "tweet_text";
# $col["width"] = "15"; // not specifying width will expand to fill space
$col["sortable"] = false; // this column is not sortable
$col["search"] = true; // this column is not searchable
$col["editable"] = false;
//$col["edittype"] = "textarea"; // render as textarea on edit
//$col["editoptions"] = array("rows"=>2, "cols"=>20); // with these attributes
$cols[] = $col;



$g = new jqgrid();

// $grid["url"] = ""; // your paramterized URL -- defaults to REQUEST_URI
$grid["rowNum"] = 20; // by default 20
$grid["sortname"] = 'id'; // by default sort grid by this field
$grid["sortorder"] = "desc"; // ASC or DESC
$grid["caption"] = "Effect Discovery"; // caption of grid
$grid["autowidth"] = true; // expand grid to screen width
$grid["multiselect"] = false; // allow you to multi-select through checkboxes

$grid["export"] = array("format"=>"xlsx", "filename"=>"my-file", "sheetname"=>"data"); //rc
//$grid["cellEdit"] = true; // inline cell editing, like spreadsheet //rc

//$g->enable_edit("FORM", "CRUD"); 

$g->set_options($grid);

$g->set_actions(array(	
						"add"=>true, // allow/disallow add
						"edit"=>true, // allow/disallow edit
						"delete"=>true, // allow/disallow delete
						"rowactions"=>true, // show/hide row wise edit/del/save option
						"search" => "advance" // show single/multi field search condition (e.g. simple or advance)
					) 
				);

// set database table for CRUD operations
$g->table = "EffectsRelationships";


// you can provide custom SQL query to display data
//$g->select_command = "SELECT * FROM EffectsRelationships";


$g->select_command = "SELECT * FROM (SELECT e.id, e.medicine, e.effect_cand, e.effect_hotness, t.sent_pos, t.sent_neg, t.sent_neu, t.sent_comp, t.tweet_id, e.root_synonym, e.stem, t.tweet_text FROM tweetsRaw t INNER JOIN EffectsRelationships e ON t.id = e.id_text_table) o";


// this db table will be used for add,edit,delete
//$g->table = "invheader";

// pass the cooked columns to grid
$g->set_columns($cols);


// generate grid output, with unique grid name as 'list1'
$out = $g->render("list1");








$themes = array("redmond","smoothness","start","dot-luv","excite-bike","flick","ui-darkness","ui-lightness","cupertino","dark-hive");
$i = rand(0,8);
?>
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.1//EN" "http://www.w3.org/TR/xhtml11/DTD/xhtml11.dtd">
<html>
<head>
	<link rel="stylesheet" type="text/css" media="screen" href="js/themes/<?php echo $themes[$i]?>/jquery-ui.custom.css"></link>	
	<link rel="stylesheet" type="text/css" media="screen" href="js/jqgrid/css/ui.jqgrid.css"></link>	
	
	<script src="js/jquery.min.js" type="text/javascript"></script>
	<script src="js/jqgrid/js/i18n/grid.locale-en.js" type="text/javascript"></script>
	<script src="js/jqgrid/js/jquery.jqGrid.min.js" type="text/javascript"></script>	
	<script src="js/themes/jquery-ui.custom.min.js" type="text/javascript"></script>
</head>
<body>
	<div style="margin:10px">

<table width="400">
     <tr>
	<td><a href="http://www.rcalix.com/clients/healthinference/phpgrid-v1.4.6/example-all.php">Health Inference</a></td>
     <td> <a href=""> Effect Discovery </a> </td>
     </tr>
     </table>

	<br>
	<?php echo $out?>
	</div>
</body>
</html>