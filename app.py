from flask import Flask, request, render_template,Response
import modules

app = Flask(__name__)

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('home.html')
            
            
@app.route('/res',methods=['POST','GET'])
def res():
    """Video streaming route. Put this in the src attribute of an img tag."""
    global result
    if request.method =='POST':
        result = request.form.to_dict()
        return render_template("results.html",result = result)

@app.route('/results')
def video_feed():
	global result
	params= result
	return Response(modules.cartoonizer(params),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True,threaded=True)
    