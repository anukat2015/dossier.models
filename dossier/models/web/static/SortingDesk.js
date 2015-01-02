/**
 * @file Sorting Desk component.
 * @copyright 2014 Diffeo
 *
 * Comments:
 * Uses the Sorting Queue component.
 *
 */


/*global SortingQueue, jQuery, define */
/*jshint laxbreak:true */


/**
 * The Sorting Desk module.
 *
 * @returns an object containing class constructors.
 * */
var SortingDesk_ = function (window, $, SortingQueue) {

  /**
   * @class
   * The main class of the Sorting Desk component.  Its responsibilities include
   * initialising the user interface and Sorting Queue, on which it depends, as
   * well as internal structures.
   *
   * Initialisation is not finalised by the time code execution exits the
   * constructor.
   *
   * @param   {Object}    opts  Initialisation options (please refer to
   *                            `defaults_' at the end of this source file)
   * @param   {Object}    cbs   Map of all callbacks
   * */
  var Sorter = function (opts, cbs)
  {
    var self = this;

    /* Allow a jQuery element to be passed in instead of an object containing
     * options. In the case that a jQuery element is detected, it is assumed to
     * be the `nodes.items' element. */
    if(!opts)
      throw "No options given: some are mandatory";
    else if(!opts.nodes)
      opts.nodes = { };

    /* Create dummy jQuery element if bins container not provided. */
    if(!opts.nodes.bins)
      opts.nodes.bins = $();

    if(!cbs)
      throw "No callbacks given: some are mandatory";

    console.log("Initialising Sorting Desk UI");

    this.options_ = $.extend(true, $.extend(true, {}, defaults_), opts);
    this.sortingQueue_ = new SortingQueue.Sorter(this.options_, cbs);

    /* Before proceeding with instance initialisation, contact the API to
     * retrieve a random item, then create a default bin based off of it and set
     * it active. */
    this.sortingQueue_.callbacks.invoke("getRandomItem")
      .done(function (result) {
        /* Ensure a `bins' options exists and that it is an instance of the
         * builtin `Array' class. Then, prepend default bin and proceed
         * with instance initialisation. */
        if(!self.options_.bins || !(self.options_.bins instanceof Array))
          self.options_.bins = [ ];

        self.options_.bins.unshift( { id: result.content_id,
                                      name: result.name } );

        /* Set query content id before initialising SortingQueue to ensure
         * correct contexts for items retrieved. */
        self.sortingQueue_.callbacks.invoke('setQueryContentId',
                                            result.content_id);
        self.initialise_();
      } )
      .fail(function (result) {
        throw "Failed to retrieve random item: " + result.error;
      } );
  };

  Sorter.prototype = {
    initialised_: false,
    options_: null,

    /* Instances */
    sortingQueue_: null,
    bins_: null,
    keyboard_: null,

    initialise_: function ()
    {
      var self = this;

      /* Begin instantiating and initialising controllers.
       *
       * Start by explicitly initialising SortingQueue's instance and proceed to
       * initialising our own instance.. */
      this.sortingQueue_.initialise();

      (this.bins_ = this.sortingQueue_.instantiate('ControllerBins', this))
        .initialise();

      (this.keyboard_ = new ControllerKeyboard(this))
        .initialise();

      this.sortingQueue_.dismiss.register('bin', function (e, id, scope) {
        var bin = self.bins_.getById(decodeURIComponent(id));

        if(bin) {
          if(bin.parent)
            bin.parent.remove(bin);
          else {
            /* Allow removal of last bin only if there is at least one item in
             * the queue. */
            if(self.bins_.bins.length == 1
               && !self.sortingQueue.items.items.length)
            {
              console.log("Disallowing removal of last bin when items' queue"
                          + " empty");
              return;
            }

            self.bins_.removeAt(self.bins_.indexOf(bin));
          }
        }
      } );

      this.options_.bins.forEach(function (descriptor) {
        var bin = self.sortingQueue_.instantiate('Bin', self.bins_, descriptor);
        self.bins_.add(bin);

        /* Instantiate and add sub-bins. */
        descriptor.children && descriptor.children.forEach(function (sb) {
          bin.add(bin.createSubBin(sb));
        } );
      } );

      this.initialised_ = true;
      console.log("Sorting Desk UI initialised");
    },

    /**
     * Resets the component to a virgin state. Removes all nodes contained by
     * `options_.nodes.bins', if any, after the active `SortingQueue' instance
     * has successfully reset.
     *
     * @returns {Promise}   Returns promise that is fulfilled upon successful
     *                      instance reset. */
    reset: function ()
    {
      if(!this.initialised_ || this.sortingQueue_.resetting())
        return this.sortingQueue_.reset();

      var self = this,
          reset = this.sortingQueue_.reset();

      reset.done(function () {
        self.keyboard_.reset();
        self.bins_.reset();

        self.bins_ = self.options_ = self.sortingQueue_ = null;
        self.initialised_ = false;

        console.log("Sorting Desk UI reset");
      } );

      return reset;
    },

    get sortingQueue ()
    { return this.sortingQueue_; },

    /**
     * Returns a boolean value indicating whether Sorting Queue has been
     * initialised and is ready to be used.
     *
     * @returns {Boolean}   Returns true if Sorting Queue has been successful
     *                      initialised, false otherwise.
     * */
    get initialised ()
    { return this.initialised_; },

    get resetting ()
    { return this.sortingQueue_.resetting(); },

    get options ()
    { return this.options_; },

    get bins ()
    { return this.bins_; },

    get keyboard ()
    { return this.keyboard_; }
  };


  /**
   * @class
   * */
  var ControllerKeyboard = function (owner)
  {
    /* Invoke super constructor. */
    SortingQueue.ControllerKeyboardBase.call(this, owner);
  };

  ControllerKeyboard.prototype = Object.create(
    SortingQueue.ControllerKeyboardBase.prototype);

  ControllerKeyboard.prototype.onKeyUp = function (evt)
  {
    var self = this,
        options = this.owner_.options;

    /* First process alpha key strokes. */
    if(evt.keyCode >= 65 && evt.keyCode <= 90) {
      var bin = this.owner_.bins.getByShortcut(evt.keyCode);

      if(this.owner_.bins.hover) {
        if(!bin)
          this.owner_.bins.setShortcut(this.owner_.bins.hover, evt.keyCode);
      } else {
        if(bin) {
          /* TODO: The following animation should be decoupled. */

          /* Simulate the effect produced by a mouse click by assigning the
           * CSS class that contains identical styles to the pseudo-class
           * :hover, and removing it after the milliseconds specified in
           * `options.delay.animateAssign'. */
          bin.node.addClass(options.css.binAnimateAssign);

          window.setTimeout(function () {
            bin.node.removeClass(options.css.binAnimateAssign);
          }, options.delays.animateAssign);

          this.owner_.sortingQueue.callbacks.invoke(
            "itemDroppedInBin",
            this.owner_.sortingQueue.items.selected(),
            bin);
          this.owner_.sortingQueue.items.remove();
        }
      }

      return false;
    }
  };


  /**
   * @class
   * */
  var ControllerBins = function (owner)
  {
    var self = this;

    /* Invoke base class constructor. */
    SortingQueue.Controller.call(this, owner);

    this.bins_ = [ ];
    this.hover_ = null;
    this.active_ = null;
    this.spawner_ = null;

    /* Define getters. */
    this.__defineGetter__("bins", function () { return this.bins_; } );
    this.__defineGetter__("hover", function () { return this.hover_; } );
    this.__defineGetter__("active", function () { return this.active_; } );
    this.__defineGetter__("node", function () {
      return this.owner_.options.nodes.bins;
    } );

    /* Instantiate spawner controller. */
    this.spawner_ = this.owner_.sortingQueue.instantiate(
      'ControllerBinSpawner',
      this,
      function (input) {
        return self.owner_.sortingQueue.instantiate(
          'Bin', self, { name: input }, null).render();
      },
      function (id, text) {
        var deferred = $.Deferred(),
            item = self.owner_.sortingQueue.items.getById(id);

        if(!item) {
          console.log("Item not found");
          deferred.reject();
        }

        window.setTimeout(function () {
          try {
            /* We rely on the API returning exactly ONE descriptor. */
            self.owner_.bins.add(
              self.owner_.sortingQueue.instantiate('Bin', self, {
                id: id,
                name: item.content.name } ) );
          } catch(x) {
            console.log("Exception occurred: " + x);
            deferred.reject();
            return;
          }

          deferred.resolve();
        }, 0);

        return deferred.promise();
      } );
  };

  ControllerBins.prototype = Object.create(SortingQueue.Controller.prototype);

  ControllerBins.prototype.initialise = function ()
  {
    this.spawner_.initialise();
  };

  ControllerBins.prototype.reset = function ()
  {
    this.spawner_.reset();
    this.node.children().remove();

    this.bins_ = this.hover_ = this.active_ = this.spawner_ = null;
  };

  ControllerBins.prototype.add = function (bin)
  {
    /* Ensure a bin with the same id isn't already contained. */
    if(this.getById(bin.id))
      throw "Bin is already contained: " + bin.id;

    /* Initialise bin. */
    bin.initialise();

    /* Contain bin and append its HTML node. */
    this.append(bin.node);
    this.bins_.push(bin);

    /* If first bin to be contained, activate it by default. */
    if(!this.active_)
      this.setActive(bin);
  };

  /* overridable */ ControllerBins.prototype.append = function (node)
  {
    /* Add bin node to the very top of the container if aren't any yet,
     * otherwise insert it after the last contained bin. */
    if(!this.bins_.length)
      this.owner_.options.nodes.bins.prepend(node);
    else
      this.bins_[this.bins_.length - 1].node.after(node);
  };

  ControllerBins.prototype.find = function (callback)
  {
    var result = null,
        search = function (bins) {
          bins.some(function (bin) {
            /* Invoke callback. A true result means a positive hit and the
             * current bin is returned, otherwise a search is initiated on the
             * children bins, if any are found. */
            if(callback(bin)) {
              result = bin;
              return true;
            } else if(bin.children.length) {
              if(search(bin.children))
                return true;
            }

            return false;
          } );
        };

    search(this.bins_);

    return result;
  };

  ControllerBins.prototype.indexOf = function (bin)
  {
    /* Note: returns the index of top level bins only. */
    return this.bins_.indexOf(bin);
  };

  ControllerBins.prototype.remove = function (bin)
  {
    return this.removeAt(this.bins_.indexOf(bin));
  };

  ControllerBins.prototype.removeAt = function (index)
  {
    var bin;

    if(index < 0 || index >= this.bins_.length)
      throw "Invalid bin index";

    bin = this.bins_[index];
    this.bins_.splice(index, 1);

    bin.node.remove();

    if(bin == this.active_)
      this.setActive(this.bins_.length && this.bins_[0] || null);
  };

  ControllerBins.prototype.setShortcut = function (bin, keyCode)
  {
    /* Ensure shortcut not currently in use and that bin is contained. */
    if(this.getByShortcut(keyCode))
      return false;

    /* Set new shortcut. */
    bin.setShortcut(keyCode);
    return true;
  };

  ControllerBins.prototype.getByShortcut = function (keyCode)
  {
    return this.find(function (bin) {
      return bin.shortcut == keyCode;
    } );
  };

  ControllerBins.prototype.getById = function (id)
  {
    return this.find(function (bin) {
      return bin.id == id;
    } );
  };

  ControllerBins.prototype.setActive = function (bin)
  {
    var self = this;

    /* Don't activate bin if currently active already. */
    if(this.active_ == bin)
      return;

    /* Invoke API to activate the bin. If successful, update UI state and force
     * a redraw of the items container. */
    if(self.active_)
      self.active_.deactivate();

    self.active_ = bin;

    if(bin) {
      bin.activate();

      if(this.owner_.initialised) {
        self.owner_.sortingQueue.callbacks.invoke("setQueryContentId", bin.id);
        self.owner_.sortingQueue.items.redraw();
      }
    }
  };

  ControllerBins.prototype.dropItem = function (bin,
                                                /* optional */ item)
  {
    this.owner_.sortingQueue.callbacks.invoke(
      "itemDroppedInBin",
      item || this.owner_.sortingQueue.items.selected(), bin);

    this.owner_.sortingQueue.items.remove();
  };

  ControllerBins.prototype.onMouseEnter_ = function (bin)
  { this.hover_ = bin; };

  ControllerBins.prototype.onMouseLeave_ = function ()
  { this.hover_ = null; };


  /**
   * @class
   * */
  var Bin = function (owner, bin)
  {
    /* Invoke super constructor. */
    SortingQueue.Drawable.call(this, owner);

    this.bin_ = bin;
    this.node_ = this.shortcut_ = null;

    this.children_ = [ ];
    this.parent_ = null;        /* Parents are set by parent bins directly. */

    /* Define getters. */
    this.__defineGetter__("bin", function () { return this.bin_; } );
    this.__defineGetter__("id", function () { return this.bin_.id; } );
    this.__defineGetter__("shortcut", function () { return this.shortcut_; } );
    this.__defineGetter__("node", function () { return this.node_; } );

    this.__defineGetter__("parent", function () { return this.parent_; } );
    this.__defineGetter__("children", function () { return this.children_; } );

    /* Define setters. */
    this.__defineSetter__("parent", function (bin) {
      if(bin.children.indexOf(this) == -1)
        throw "Not a parent of bin";

      this.parent_ = bin;
    } );
  };

  Bin.prototype = Object.create(SortingQueue.Drawable.prototype);

  Bin.prototype.initialise = function ()
  {
    var self = this,
        parentOwner = self.owner_.owner;

    (this.node_ = this.render())
      .attr( {
        'data-scope': 'bin',
        'id': encodeURIComponent(this.id)
      } )
      .on( {
        mouseenter: function () {
          self.owner_.onMouseEnter_(self);
          return false;
        },
        mouseleave: function () {
          self.owner_.onMouseLeave_();
          return false;
        }
      } );

    new SortingQueue.Droppable(this.node_, {
      classHover: parentOwner.options.css.droppableHover,
      scopes: [ 'text-item', 'bin' ],

      drop: function (e, id, scope) {
        id = decodeURIComponent(e.dataTransfer.getData('DossierId'));

        switch(scope) {
        case 'text-item':
          var item = parentOwner.sortingQueue.items.getById(id);

          parentOwner.sortingQueue.callbacks.invoke("itemDroppedInBin",
                                                    item,
                                                    self);

          parentOwner.sortingQueue.items.remove(item);
          break;

        case 'bin':
          var bin = self.owner_.getById(id);

          parentOwner.sortingQueue.callbacks.invoke("mergeBins",
                                                    self,
                                                    bin);

          self.owner_.removeAt(self.owner_.indexOf(bin));

          /* Important: DOM node is destroyed above, which means the `dragend'
           * event won't be triggered, leaving the dismissal button visible. */
          parentOwner.sortingQueue.dismiss.deactivate();

          break;

        default:
          throw "Invalid scope: " + scope;
        }
      }
    } );

    /* We must defer initialisation of D'n'D because owning object's `bin'
     * attribute will have not yet been set. */
    window.setTimeout(function () {
      new SortingQueue.Draggable(self.node_, {
        dragstart: function (e) {
          parentOwner.sortingQueue.dismiss.activate();
        },

        dragend: function (e) {
          parentOwner.sortingQueue.dismiss.deactivate();
        }
      } );
    }, 0);
  };

  /* overridable */ Bin.prototype.createSubBin = function (bin)
  {
    return this.owner_.owner.sortingQueue.instantiate('Bin', this.owner_, bin);
  };

  Bin.prototype.add = function (bin)
  {
    /* Ensure a bin with the same id isn't already contained. */
    if(this.owner_.getById(bin.id))
      throw "Bin is already contained: " + bin.id;

    this.children_.push(bin);

    /* Initialise bin and append its HTML. */
    bin.parent = this;
    bin.initialise();

    this.append(bin.node);
  };

  Bin.prototype.activate = function ()
  {
    this.node.addClass(this.owner_.owner.options.css.binActive);
  };

  Bin.prototype.deactivate = function ()
  {
    this.node.removeClass(this.owner_.owner.options.css.binActive);
  };

  Bin.prototype.indexOf = function (bin)
  {
    /* Note: returns the index of immediately contained children bins. */
    return this.children_.indexOf(bin);
  };

  Bin.prototype.remove = function (bin)
  {
    return this.removeAt(this.children_.indexOf(bin));
  };

  Bin.prototype.removeAt = function (index)
  {
    var bin;

    if(index < 0 || index >= this.children_.length)
      throw "Invalid bin index";

    bin = this.children_[index];
    this.children_.splice(index, 1);

    bin.node.remove();
  };

  /* overridable */ Bin.prototype.append = function (node)
  {
    this.getNodeChildren().append(node);
  };

  Bin.prototype.setShortcut = function (keyCode)
  {
    this.shortcut_ = keyCode;

    /* Set shortcut visual cue, if a node exists for this purpose. */
    var node = this.getNodeShortcut();
    if(node && node.length)
      node[0].innerHTML = String.fromCharCode(keyCode).toLowerCase();
  };

  /* overridable */ Bin.prototype.getNodeShortcut = function ()
  {
    return this.node_.find('.' + this.owner_.owner.options.css.binShortcut);
  };

  /* overridable */ Bin.prototype.getNodeChildren = function ()
  {
    return this.node_.find('>.'
                           + this.owner_.owner.options.css.binChildren
                           + ':nth(0)');
  };


  /**
   * @class
   * */
  var BinDefault = function (owner, bin)
  {
    /* Invoke super constructor. */
    Bin.call(this, owner, bin);
  };

  BinDefault.prototype = Object.create(Bin.prototype);

  BinDefault.prototype.initialise = function ()
  {
    var self = this;

    /* Invoke base class implementation. */
    Bin.prototype.initialise.call(this);

    /* Now add an additional event handler for mouse clicks, which will, by
     * default, invoke the item dropped event handler in `ControllerBin'. */
    this.node_.click(function () {
      self.owner_.setActive(self);
      return false;
    } );

  };

  BinDefault.prototype.render = function ()
  {
    var css = this.owner_.owner.options.css;

    return $('<div class="' + css.binTop + '"><div class="'
             + css.binShortcut + '"/><div>' + this.bin_.name
             + '</div><div class="' + css.binChildren + '"></div></div>');
  };


  /**
   * @class
   * */
  var ControllerBinSpawner = function (owner, fnRender, fnAdd)
  {
    /* Invoke super constructor. */
    SortingQueue.Controller.call(this, owner);

    this.fnRender_ = fnRender;
    this.fnAdd_ = fnAdd;
  };

  ControllerBinSpawner.prototype =
    Object.create(SortingQueue.Controller.prototype);

  ControllerBinSpawner.prototype.reset = function ()
  {
    this.fnRender_ = this.fnAdd_ = null;
  };

  ControllerBinSpawner.prototype.add = function (id)
  {
    var parentOwner = this.owner_.owner,
        options = parentOwner.options;

    /* Do not allow entering into concurrent `add' states. */
    if(this.owner_.node.find('.' + options.css.binAdding).length)
      return;

    var nodeContent = id ? 'Please wait...'
          : ('<input placeholder="Enter bin description" '
             + 'type="text"/>'),
        node = this.fnRender_(nodeContent)
          .addClass(options.css.binAdding)
          .fadeIn(options.delays.addBinShow);

    this.owner_.append(node);

    if(!id) {
      /* TODO: manually creating a bin is DISABLED as this implementation
       * currently has no way of obtaining a label (search engine) for the
       * bin. */
      console.log("Manually creating a bin is DISABLED");
/*       this.addManual(node); */
      return;
    }

    var item = parentOwner.sortingQueue.items.getById(id);

    if(!item) {
      node.remove();
      throw "add: failed to retrieve text item: " + id;
    }

    this.fnAdd_(id, item.content.text)
      .always(function () { node.remove(); } );
  };

  ControllerBinSpawner.prototype.addManual = function (node)
  {
    /* TODO: manually creating a bin is DISABLED as this implementation
     * currently has no way of obtaining a label (search engine) for the
     * bin. */
    throw "Manually creating a bin is DISABLED";

    var self = this,
        input = node.find('input');

    input
      .focus()
      .blur(function () {
        if(!this.value) {
          node.fadeOut(self.owner_.owner.options.delays.addBinShow,
                       function () { node.remove(); } );
          return;
        }

        this.disabled = true;

        /* TODO: do not use an `alert'. */
        self.fnAdd_(null, this.value)
          .fail(function () { alert("Failed to create a sub-bin."); } )
          .always(function () { node.remove(); } );
      } )
      .keyup(function (evt) {
        if(evt.keyCode == 13)
          this.blur();

        /* Do not allow event to propagate. */
        return false;
      } );
  };


  /**
   * @class
   * */
  var ControllerBinSpawnerDefault = function (owner, fnRender, fnAdd)
  {
    /* Invoke base class constructor. */
    ControllerBinSpawner.call(this, owner, fnRender, fnAdd);

    this.node_ = null;
  };

  ControllerBinSpawnerDefault.prototype =
    Object.create(ControllerBinSpawner.prototype);

  ControllerBinSpawnerDefault.prototype.initialise = function ()
  {
    var self = this,
        parentOwner = this.owner_.owner;

    this.node_ = this.render()
      .addClass(parentOwner.options.css.buttonAdd)
      .on( {
        click: function () {
          self.add();
        }
      } );

    new SortingQueue.Droppable(this.node_, {
      classHover: parentOwner.options.css.droppableHover,
      scopes: [ 'text-item' ],

      drop: function (e, id) {
        id = decodeURIComponent(id);

        self.add(id);

        var items = self.owner_.owner.sortingQueue.items;
        items.remove(items.getById(id));
      }
    } );

    this.owner_.node.append(this.node_);
  };

  ControllerBinSpawnerDefault.prototype.reset = function ()
  {
    if(this.node_) {
      this.node_.remove();
      this.node_ = null;
    }

    ControllerBinSpawner.prototype.reset.call(this);
  };

  /* overridable */ ControllerBinSpawnerDefault.prototype.render = function ()
  {
    return $('<div><span>+</span></div>');
  };


  var defaults_ = {
    css: {
      binTop: 'sd-bin',
      binShortcut: 'sd-bin-shortcut',
      binChildren: 'sd-children',
      binAnimateAssign: 'sd-assign',
      binAdding: 'sd-adding',
      binActive: 'sd-active',
      buttonAdd: 'sd-button-add',
      droppableHover: 'sd-droppable-hover'
    },
    delays: {                   /* In milliseconds.     */
      animateAssign: 75,        /* Duration of assignment of text item via
                                 * shortcut. */
      binRemoval: 200,          /* Bin is removed from container. */
      addBinShow: 200           /* Fade in of temporary bin when adding. */
    },
    constructors: {
      ControllerBins: ControllerBins,
      Bin: BinDefault,
      ControllerBinSpawner: ControllerBinSpawnerDefault
    }
  };

  return {
    Sorter: Sorter,
    ControllerBins: ControllerBins,
    Bin: Bin,
    BinDefault: BinDefault,
    ControllerBinSpawner: ControllerBinSpawner,
    ControllerBinSpawnerDefault: ControllerBinSpawnerDefault
  };

};


/* Compatibility with RequireJs. */
if(typeof define === "function" && define.amd) {
  define("SortingDesk", [ "jquery", "SortingQueue" ],
         function ($, SortingQueue) {
           return SortingDesk_(window, $, SortingQueue);
         } );
} else
  window.SortingDesk = SortingDesk_(window, jQuery, SortingQueue);
