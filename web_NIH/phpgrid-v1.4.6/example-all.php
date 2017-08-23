<?php 
/**
 * PHP Grid Component
 *
 * @author Abu Ghufran <gridphp@gmail.com> - http://www.phpgrid.org
 * @version 1.4.6
 * @license: see license.txt included in package
 */

$conn = mysql_connect(Òip", ÒuserÓ, ÒpassÓ);
mysql_select_db(Òdb_nameÓ);

// set your db encoding -- for ascent chars (if required)
mysql_query("SET NAMES 'utf8'");

include("inc/jqgrid_dist.php");

// you can customize your own columns ...

$col = array();
$col["title"] = "Id"; // caption of column
$col["name"] = "id"; // grid column name, must be exactly same as returned column-name from sql (tablefield or field-alias) 
$col["width"] = "15";
$cols[] = $col;		

$col = array();
$col["title"] = "tweet_id";
$col["name"] = "tweet_id"; 
$col["width"] = "40";
$col["editable"] = false; // this column is editable
$col["editoptions"] = array("size"=>20); // with default display of textbox with size 20
$col["editrules"] = array("required"=>true, "edithidden"=>true); // and is required
$col["hidden"] = false;
// $col["formatter"] = "date"; // format as date
// $col["formatoptions"] = array("srcformat"=>'Y-m-d',"newformat"=>'d/m/Y'); // @todo: format as date, not working with editing

$cols[] = $col;
		
$col = array();
$col["title"] = "created_at";
$col["name"] = "created_at";
$col["width"] = "40";
$col["sortable"] = true; 
$col["editable"] = false; // this column is not editable
$col["align"] = "center"; // this column is not editable
$col["search"] = true; // this column is not searchable
//$col["formatter"] = "date"; //"image"; // format as image -- if data is image url e.g. http://<domain>/test.jpg
//$col["formatoptions"] = array("srcformat"=>'Y-m-d',"newformat"=>'d/m/Y'); // image width / height etc

$cols[] = $col;

$col = array();
$col["title"] = "tweet";
$col["name"] = "tweet_text";
# $col["width"] = "300"; // not specifying width will expand to fill space
$col["sortable"] = false; // this column is not sortable
$col["search"] = true; // this column is not searchable
$col["editable"] = false;
$col["edittype"] = "textarea"; // render as textarea on edit
$col["editoptions"] = array("rows"=>2, "cols"=>20); // with these attributes
$cols[] = $col;

$col = array();
$col["title"] = "retweets";
$col["name"] = "retweets";
$col["width"] = "15";
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
$col["title"] = "favorites";
$col["name"] = "favorite_count";
$col["width"] = "15";
$col["editable"] = false;

// default render is textbox
// $col["editoptions"] = array("value"=>'10');

// can be switched to select (dropdown)
# $col["edittype"] = "select"; // render as select
# $col["editoptions"] = array("value"=>'10:$10;20:$20;30:$30;40:$40;50:$50'); // with these values "key:value;key:value;key:value"

$cols[] = $col;


$col = array();
$col["title"] = "pet";
$col["name"] = "pet_class";
$col["width"] = "15";
$col["editable"] = false;

// default render is textbox
// $col["editoptions"] = array("value"=>'10');

$cols[] = $col;


$col = array();
$col["title"] = "annot";
$col["name"] = "annotation";
$col["width"] = "15";
$col["editable"] = true;
#$col["edittype"] = "checkbox"; // render as checkbox
#$col["editoptions"] = array("value"=>"yes:no"); // with these values "checked_value:unchecked_value"
$col["edittype"] = "select"; // render as select
$col["editoptions"] = array("value"=>'no:no;yes:yes'); 


// default render is textbox
// $col["editoptions"] = array("value"=>'10');

$cols[] = $col;



$col = array();
$col["title"] = "neg";
$col["name"] = "sent_neg";
$col["width"] = "15";
$col["editable"] = false;
$col["edittype"] = "checkbox"; // render as checkbox
$col["editoptions"] = array("value"=>"1:0"); // with these values "checked_value:unchecked_value"
#$col["edittype"] = "select"; // render as select
#$col["editoptions"] = array("value"=>'No:Not Booked;Yes:Yes it is Booked'); 
$cols[] = $col;

$col = array();
$col["title"] = "pos";
$col["name"] = "sent_pos";
$col["width"] = "15";
$col["editable"] = false;
$col["edittype"] = "checkbox"; // render as checkbox
$col["editoptions"] = array("value"=>"1:0"); // with these values "checked_value:unchecked_value"
#$col["edittype"] = "select"; // render as select
#$col["editoptions"] = array("value"=>'No:Not Booked eg. Ñ, Í,É;Yes:Yes it is Booked eg. Ñ, Í,É'); 
$cols[] = $col;


$col = array();
$col["title"] = "neu";
$col["name"] = "sent_neu";
$col["width"] = "15";
$col["editable"] = false;
$col["edittype"] = "checkbox"; // render as checkbox
$col["editoptions"] = array("value"=>"1:0"); // with these values "checked_value:unchecked_value"
#$col["edittype"] = "select"; // render as select
#$col["editoptions"] = array("value"=>'No:Not Booked eg. Ñ, Í,É;Yes:Yes it is Booked eg. Ñ, Í,É'); 
$cols[] = $col;


$col = array();
$col["title"] = "eff_extract";
$col["name"] = "effects_extracted";
$col["width"] = "15";
$col["editable"] = true;
//$col["edittype"] = "checkbox"; // render as checkbox
//$col["editoptions"] = array("value"=>"1:0"); // with these values "checked_value:unchecked_value"
#$col["edittype"] = "select"; // render as select
#$col["editoptions"] = array("value"=>'No:Not Booked eg. Ñ, Í,É;Yes:Yes it is Booked eg. Ñ, Í,É'); 
$cols[] = $col;



$g = new jqgrid();

// $grid["url"] = ""; // your paramterized URL -- defaults to REQUEST_URI
$grid["rowNum"] = 20; // by default 20
$grid["sortname"] = 'id'; // by default sort grid by this field
$grid["sortorder"] = "desc"; // ASC or DESC
$grid["caption"] = "Health inference"; // caption of grid
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
$g->table = "tweetsRaw";


// you can provide custom SQL query to display data
$g->select_command = "SELECT * FROM tweetsRaw";

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
	<td>Health Inference</td><td> <a href="http://www.rcalix.com/clients/healthinference/phpgrid-v1.4.6/effectDiscovery.php"> Effect Discovery </a> </td>
     </tr>
     </table>

	<br>
	<?php echo $out?>
	</div>
</body>
</html>