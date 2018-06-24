COUPLET_URL = "couplet"
$(document).ready(function(){
	$('#couplet-input').bind('input propertychange', function() {  
		input = $('#couplet-input').val();
		$('#inseq').empty();
		for(i=0;i<input.length;i++){
		    if(input[i]==" ")
		        continue;
			$('#inseq').append("<span>"+input[i]+"</span>");
		}
	});
	$('#input-btn').click(function(){
		input = $('#couplet-input').val();
		input_seq = "";
		for(i=0;i<input.length;i++){
            if(input[i]==" ")
                    continue;
			input_seq += input[i] + " ";
		}
		if(input_seq.length<10||input_seq.length>20){
			alert('建议输入序列在5-10个词');
			return;
		}
		
		$.ajax({
			url : COUPLET_URL,
			type : "get",
			async : false,
			data : {
				inseq : input_seq
			},
			success : function(data) {
			    data = $.parseJSON(data)
				out = data['outseq'];
				$("#outseq").empty();
				for(i=0;i<out.length;i++){
					$('#outseq').append("<span>"+out[i]+"</span>");
				}
			}
		});
	});
});